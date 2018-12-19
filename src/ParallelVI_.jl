module ParallelVI_

export ParallelVI, solve, solve2, solveChunk, solveChunkTest, parallelTest, unwrap_deg, unwrap_rad

include("CSLVProblem_.jl")
using Distributed
using SharedArrays
using GridInterpolations

######  parallel value iteration functions  ######

mutable struct ParallelVI
    policyFile::String
    maxIterations::Int
    tolerance::Float64
    nProcs::Int
    valU::Vector
    valQ::Matrix
    stateOrder::Vector
end

function solve(mdp::CSLVProblem_.CSLVProblem, alg::ParallelVI)
    nStates  = mdp.nStates
    nActions = mdp.nActions
    maxIter = alg.maxIterations
    tol     = alg.tolerance
    valU = alg.valU
    valQ = alg.valQ
    nProcs = alg.nProcs
    nChunks = length(alg.stateOrder)
    order = alg.stateOrder

    if nProcs > Sys.CPU_THREADS
        error("Requested too many processors")
    end

    # start and end indeces
    chunks = Array{Vector{Tuple{Int64, Int64}}}(undef,nChunks) 
    for i = 1:nChunks
        co = order[i]
        sIdx = co[1]
        eIdx = co[2]
        ns = eIdx - sIdx
        # divide the work among the processors
        stride = div(ns,(nProcs-1))
        temp = Tuple{Int64, Int64}[]
        for j = 0:(nProcs-2)
            si = j * stride + sIdx
            ei = si + stride - 1
            if j == (nProcs-2) 
                ei = eIdx
            end
	push!(temp, (si ,ei))
        end
	chunks[i] = temp
    end

    # shared array for utility
    init_qval = zeros(nStates, nActions)
    util = SharedArray{Float64}((nStates), init = S -> S[localindices(S)] .= 0.0, pids = 1:nProcs)
    qValues  = SharedArray{Float64}((nStates, nActions), init = S -> S[localindices(S)] = init_qval[localindices(S)], pids = 1:nProcs)

    # loop over chunks 
    results = 1
    uCount = 0
    for i = 1:maxIter
        # utility array update indeces 
        for c = 1:nChunks
            lst = chunks[c]

            uIdx1 = uCount % 2 + 1
            uIdx2 = (uCount+1) % 2 + 1

            # update q-val only on the last iteration
            println("Chunk: $c start:")
            @time results = pmap(x -> solveChunk(c, mdp, util, qValues, x), lst) 
            @time mdp = updateMDP(mdp, c, util, results)
            println("Chunk: $c end")

            uCount += 1
        end # chunk loop 

    end # main iteration loop
    return util, qValues
end

function solveChunk(c::Int64, mdp::CSLVProblem_.CSLVProblem, util::SharedArray, qValues::SharedArray, stateIndices::Tuple{Int64, Int64})
    sStart = stateIndices[1]
    sEnd   = stateIndices[2]
    nActions = mdp.nActions

    minX = Inf
    maxX = -Inf
    minY = Inf
    maxY = -Inf

    for si = sStart:sEnd
        qHi = -Inf
        ai = 0

        s = CSLVProblem_.State(GridInterpolations.ind2x(mdp.acGrid[c], si))

        for a in mdp.actionArray
            ai += 1
            probs, states = nextState(mdp, s, a) 
            qNow = reward(mdp, s, a)

            for sp = 1:length(states)
                x = states[sp]
                # mdp.acGrid[c-1] because nextState grid
                if c > 1
                    qNow += probs[sp] * GridInterpolations.interpolate(mdp.acGrid[c-1], util, x) 
                end
            end # sp loop
            qValues[si,ai] = qNow
            if ai == 1 
                qHi = qNow
                util[si] = qHi
        
                if qHi != 0 
                    if s.x < minX
                        minX = s.x
                    elseif s.x > maxX
                        maxX = s.x
                    end
                    if s.y < minY
                        minY = s.y
                    elseif s.y > maxY
                        maxY = s.y
                    end
                end

            elseif qNow > qHi

                qHi = qNow
                util[si] = qHi
                if s.x < minX 
                    minX = s.x
        elseif s.x > maxX
                    maxX = s.x
                end
                if s.y < minY
                    minY = s.y
                elseif s.y > maxY 
                    maxY = s.y
                end
            end
        end # action loop
    end # state loop
    return sStart, sEnd, minX, maxX, minY, maxY
end

function parallelTest(nProcs::Int, x::Float64, y::Float64, at::Float64, pt::Float64; nIter::Int=1, nChunks::Int =82)
    problem = CSLVProblem_.CSLVProblem(x, y, at, pt)

    nStates  = prod(problem.acGrid[1].cut_counts)
    nActions = length(problem.actionArray) 

    order = Array{Vector{Int}}(undef,nChunks)
    stride = div(nStates,nChunks)
    for i = 0:(nChunks-1)
        sIdx = i * stride + 1
        eIdx = sIdx + stride - 1
        if i == (nChunks-1) && eIdx != nStates
            eIdx = nStates
        end
        order[i+1] = [sIdx, eIdx] 

    end

    pvi = ParallelVI("pvi_policy.pol", nIter, 1e-4, nProcs, zeros(2), zeros(2,2), order) 

    @time utils, qvalues = ParallelVI_.solve(problem, pvi)

    return utils, qvalues, problem
end

function updateMDP(mdp::CSLVProblem_.CSLVProblem, c::Int64, util::SharedArray, results::Array{Tuple{Int64,Int64,Float64,Float64,Float64,Float64},1})
    r = [y[i] for y in results, i in 1:length(results[1])]
    r3 = minimum(r[:,3])
    r4 = maximum(r[:,4])
    r5 = minimum(r[:,5])
    r6 = maximum(r[:,6])
    if r3 == Inf
        mdp.minE = mdp.minErecord[1]
    else
        mdp.minE = r3 - mdp.distanceBuffer
    end
    if r4 == -Inf
        mdp.maxE = mdp.maxErecord[1]
    else
        mdp.maxE = r4 + mdp.distanceBuffer
    end
    if r5 == Inf
        mdp.minN = mdp.minNrecord[1]
    else 
        mdp.minN = r5 - mdp.distanceBuffer
    end
    if r6 == -Inf
        mdp.maxN = mdp.maxNrecord[1] 
    else
        mdp.maxN = r6 + mdp.distanceBuffer
    end

    mdp.minErecord[c+1] = mdp.minE
    mdp.maxErecord[c+1] = mdp.maxE
    mdp.minNrecord[c+1] = mdp.minN
    mdp.maxNrecord[c+1] = mdp.maxN

    push!(mdp.acGrid, RectangleGrid(collect(range(mdp.minE,stop=mdp.maxE,length=mdp.distanceDivision)), 
                         collect(range(mdp.minN,stop=mdp.maxN,length=mdp.distanceDivision)),                
                         -180.:mdp.stepHeadingState:180., -1.:mdp.timeStep:mdp.lvTime, 
                         0.:mdp.timeStep:mdp.timeThres))
    return mdp
end

######  value iteration functions   ######

function nextState(problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State, action::CSLVProblem_.Action)
  ## if first time step empty
  if state.timeRem == 0.
    return ((0.,0.),(zeros(5),zeros(5)))
  end
  ## based on action, possible responses and headings
  if action.head != problem.noAlert
    ## does not respond, does respond
    response = [1-problem.response,problem.response]
    if problem.restricted
        headings = [unwrap_deg(state.head), unwrap_deg(state.head + action.head)]
    else    
        headings = [unwrap_deg(state.head), action.head]
    end
  else
    ## turning probability distribution
    response = problem.turnDist
    headings = [unwrap_deg(state.head-2*problem.stepHeadingState), unwrap_deg(state.head-problem.stepHeadingState), 
                unwrap_deg(state.head), unwrap_deg(state.head+problem.stepHeadingState), 
                unwrap_deg(state.head+2*problem.stepHeadingState)]
  end
  ## setup arrays for valid speeds and positions
  xSpeeds = Array{Float64}(undef,length(headings))
  ySpeeds = Array{Float64}(undef,length(headings))
  possibleEast = Array{Float64}(undef,length(headings))
  possibleNorth = Array{Float64}(undef,length(headings))

  ## setup the speeds and positions
  for i = 1:length(headings)
    xSpeeds[i] = round(Int64,problem.acSpeed*cosd(headings[i]))
    ySpeeds[i] = round(Int64,problem.acSpeed*sind(headings[i]))
    possibleEast[i] = state.x + xSpeeds[i]
    possibleNorth[i] = state.y + ySpeeds[i]
    ## check within bounds and set to limits if outside
    trIdx = round(Int64,state.timeRem)
    if possibleEast[i] <= problem.minErecord[trIdx] || possibleEast[i] >= problem.maxErecord[trIdx]
      return ((0.,0.),(zeros(5),zeros(5)))
    elseif possibleNorth[i] <= problem.minNrecord[trIdx] || possibleNorth[i] >= problem.maxNrecord[trIdx]
      return ((0.,0.),(zeros(5),zeros(5)))
    end
  end
  ## next states if anomaly already occurred 
  if state.anomaly >= 0.
    ## only next states with anomaly 
    nextStates = Array{Array}(undef,length(headings))
    for i=1:length(headings)
      nextStates[i] = [possibleEast[i], possibleNorth[i], headings[i], state.anomaly, state.timeRem-1.]
    end
    return ((response),(nextStates))
  ## next state if anomaly has not occurred
  else
    nextStatesNoAnom = Array{Array}(undef,length(headings))
    nextStatesAnom = Array{Array}(undef,length(headings))
    for i = 1:length(problem.lvStates)
      ## check if anomaly can occur and setup potential next states
      if approxEq(problem.lvStates[i].timeLV, (state.timeRem-problem.timeThres))
        for j = 1:length(headings)
          nextStatesNoAnom[j] = [possibleEast[j], possibleNorth[j], headings[j], -1, state.timeRem-1.]
          nextStatesAnom[j] = [possibleEast[j], possibleNorth[j], headings[j], problem.lvStates[i].timeLV, state.timeRem-1.]
        end
        nextStates = Array{Array}(undef,length(headings)*2)
        probResponse = Array{Float64}(undef,length(headings)*2)
        for j = 1:length(headings)
          if action.head != problem.noAlert
            nextStates[j] = nextStatesNoAnom[j]
            nextStates[j+length(headings)] = nextStatesAnom[j]
            probResponse[j] = response[1]*(1.0-problem.lvStates[i].probAnom)
            probResponse[j+length(headings)] = response[2]*(problem.lvStates[i].probAnom)
          else
            nextStates[2*j-1] = nextStatesNoAnom[j]
            nextStates[2*j] = nextStatesAnom[j]
            probResponse[2*j-1] = response[j]*(1.0-problem.lvStates[i].probAnom)
            probResponse[2*j] = response[j]*(problem.lvStates[i].probAnom)
          end
        end
        return ((probResponse), (nextStates))
      end
    end
    ## anomaly cannot occur
    for i=1:length(headings)
      nextStatesNoAnom[i] = [possibleEast[i], possibleNorth[i], headings[i], -1, state.timeRem-1.]
    end
    return ((response),(nextStatesNoAnom))
  end
end

## reward is composed of velocity and distance reward
function reward(problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State, action::CSLVProblem_.Action)
  if velocityReward(problem, state, action) == -Inf
    return -Inf
  else
    ## use lambda to weight safety vs. efficiency
    return problem.lambda * velocityReward(problem, state, action) + distanceReward(problem, state)[1]
  end
end

## setup the velocity reward that depends on heading change
function velocityReward(problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State, action::CSLVProblem_.Action)
  ## no alert
  if action.head == problem.noAlert
    return 0.
  ## weak left or right turn
  elseif abs(action.head) == problem.stepHeadingState
    return -0.5
  ## strong left or right turn
  elseif abs(action.head) == 2*problem.stepHeadingState
    return -1.
  else
    return problem.maintainCost ## reward if alert commands current heading
  end
end

## setup the distance reward that depends on state and calls debris locations
function distanceReward(problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State)
  ## not within safety threshold of debris
  reward, ellipse = 0., 0.
  ## if anomaly, determine if aircraft is at risk from any debris
  if state.anomaly >= 0.
    reward, ellipse = debrisLocations(problem, state)
    reward = -reward
  end
  reward, ellipse
end

## find the debris and call inEllipse to determine if there is a penalty
function debrisLocations(problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State)
  ## current location
  point = [state.x,state.y]
  if state.anomaly == problem.debris.anomalyTime - 1 || state.anomaly == problem.debris.anomalyTime || state.anomaly == problem.debris.anomalyTime + 1
    if state.timeRem == problem.debris.remainingTime - 2 || state.timeRem == problem.debris.remainingTime - 1 || state.timeRem == problem.debris.remainingTime || state.timeRem == problem.debris.remainingTime + 1 || state.timeRem == problem.debris.remainingTime + 2
      hitD, inE = inEllipse(problem, state, problem.debris.eLocation, problem.debris.nLocation)
      if inE == 1.
        return 1., 1. # returnValue, ellipseStatus
      else
        return 0., 0. # returnValue, ellipseStatus
      end
    end
  end
  return 0., 0. # no active debris
end

## find debris ellipse and determine if aircraft within debris ellipse
function inEllipse(problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State, cX::Float64, cY::Float64)
  ## find current lv heading (direction of debris ellipse major axis)
  a = problem.safeThres
  ## check if aircraft is within debris ellipse
  curDist = sqrt((state.x-cX)^2+(state.y-cY)^2)
  if curDist <= a
    return 1., 1.
  else
    return exp(-2(curDist-a)/a), 0.
  end
end

######  general helper functions    ######

function approxEq(x,y,epsilon = 1.e-2)
  abs(x-y) < epsilon
end

## use this function to wrap angles when in radians
function unwrap_rad(a_rad::Float64)
  if (a_rad > pi)
      return mod(a_rad + pi, 2*pi) - pi
  elseif (a_rad < -pi)
      return -(mod(-a_rad + pi, 2*pi) - pi)
  else
      return a_rad
  end
end

## use this function to wrap angles when in degrees
function unwrap_deg(a_deg::Float64)
  unwrap_rad(a_deg*pi/180)*180/pi
end

end # ParallelVI_ module
