module CSLVProblem_

using GridInterpolations, NearestNeighbors

import Base.convert

export State, convert, LVState, Action, DebrisPeice, CSLVProblem

## for problem.acGrid
struct State
  x::Float64
  y::Float64
  head::Float64
  anomaly::Float64
  timeRem::Float64
end
convert(s::State) = [s.x, s.y, s.head, s.anomaly, s.timeRem]
State(x::Vector{Float64}) = State(x...)

## for problem.lvStates
struct LVState
  x::Float64
  y::Float64
  probAnom::Float64
  safe::Float64
  timeLV::Float64
end

## for problem.actionArray
struct Action
  head::Float64
end

mutable struct DebrisPeice
  eLocation::Float64
  nLocation::Float64
  anomalyTime::Float64
  remainingTime::Float64
end

## to fully define problem and all fields
mutable struct CSLVProblem
  minE::Float64
  maxE::Float64
  minN::Float64
  maxN::Float64
  stepHeadingState::Float64
  timeThres::Int64
  acGrid::Array{RectangleGrid}
  lvStates::Vector{LVState}
  noAlert::Float64
  actionArray::Array{Action}
  debris::DebrisPeice
  nStates::Int64
  nActions::Int64
  maintainCost::Float64
  headingLimit::Float64
  startDebris::Float64
  endDebris::Float64
  timeStepSeconds::Float64
  intersectTime::Float64
  lambda::Float64
  acSpeed::Float64
  response::Float64
  turnDist::Array{Float64}
  train::Bool
  restricted::Bool
  oldDebris::Bool
  minErecord::Array{Float64}
  maxErecord::Array{Float64}
  minNrecord::Array{Float64}
  maxNrecord::Array{Float64}
  distanceDivision::Int64
  distanceBuffer::Float64
  timeStep::Float64
  lvTime::Float64
  safeThres::Float64
end

## MAKE the problem
function CSLVProblem(x::Float64, y::Float64, at::Float64, pt::Float64;restricted=true, train=true, oldDebris=false)
  
  ## setup bank of debris profiles to cycle through  
  startDebrisNumber = 1
  endDebrisNumber = 25
  startDebris = 4;
  endDebris = 10;  
    
  #############################################################
  ##          Parameters needed for immutable State          ##
  #############################################################

  timeStep = 1. # * 10 seconds (simulation time increments)

  timeThres = 81 # * 10 seconds (total simulation time)

  debris_x = x # 2361. # 4542.
  debris_y = y # -5321. # -4737.
   
  debris = DebrisPeice(x, y, at, timeThres-pt)

  # make debris st

  acSpeed = 2900. # m/10seconds
  # actual not inflated
  safeThres = 152. * 10 # meters (rounded)

  distanceDivision = 11 # 45 # 11 # 10 # 50
  distanceBuffer = acSpeed*2 + safeThres*2

  minErecord = zeros(timeThres+1+1) # +1 for 0 +1 for initial
  maxErecord = zeros(timeThres+1+1) # +1 for 0 +1 for initial
  minNrecord = zeros(timeThres+1+1) # +1 for 0 +1 for initial
  maxNrecord = zeros(timeThres+1+1) # +1 for 0 +1 for initial

  ## For Grid: East position, North position, Heading, 
  ## Anomaly Time, Launch Vehicle Time ##

  ## E position parameters all in meters
  minE = debris_x - distanceBuffer
  maxE = debris_x + distanceBuffer
 
  ## N position parameters all in meters
  minN = debris_y - distanceBuffer
  maxN = debris_y + distanceBuffer 

  minErecord[1] = minE
  maxErecord[1] = maxE
  minNrecord[1] = minN
  maxNrecord[1] = maxN

  ## heading change increments  
  stepHeadingState = 15. # degrees

  lvTime = 11. # * 10 seconds (total launch vehicle simulation time)

  ## make acGrid
  acGrid = [RectangleGrid(collect(range(minE,stop=maxE,length=distanceDivision)), collect(range(minN,stop=maxN,length=distanceDivision)),
                          -180.:stepHeadingState:180., -1.:timeStep:lvTime, 0.:timeStep:timeThres)]
  
  #############################################################
  ##        Parameters needed for immutable LVState          ##
  #############################################################

  ## launch vehicle trajectory ENU from cape canaveral, 1 position for every 10 seconds
  eUse = [0.0, 0.0, -0.0, 0.0, 0.0, 107.82, 1147.6, 3484.6, 7397.02, 13321.5, 21715.8, 33305.8]
  nUse = [0.0, -0.0, -11.0859, -11.0899, -11.0955, 44.4101, 722.251, 2269.08, 4911.38, 8967.77,
         14806.3, 23015.7]

  # probability of anomaly for each time step
  probAnom = 0.052 

  ## MAKE lvStates
  prob = fill(0.052,length(eUse))
  safe = fill(safeThres,length(eUse))
  timeLV = collect(0.:timeStep:lvTime)
  # setup LV states with timeRem
  lvStates = [LVState(eUse[i], nUse[i], prob[i], safe[i], timeLV[i]) for i in 1:length(eUse)]

  #############################################################
  ##         Parameters needed for immutable Action          ##
  #############################################################

  ## value action is set to when no alert is issued
  noAlert = 3000.

  stepHeadingAction = 15. # degrees

  ## MAKE actionArray
  heading = [noAlert, -2*stepHeadingAction, -stepHeadingAction, 0, stepHeadingAction, 2*stepHeadingAction]
  actionArray = [Action(heading[i]) for i = 1:length(heading)]

  #############################################################
  ## Parameters needed for type CSLVProblem ##
  #############################################################

  ## for parallelization and function: valueIteration
  nStates = prod(acGrid[1].cut_counts)
  nActions = length(actionArray)

  ## function: velocityReward 
  maintainCost = -0.01
  headingLimit = 30.1 # degrees

  ## function: debrisLocations
  timeStepSeconds = 10.

  ## function: distanceReward
  intersectTime = 76. # * 10 seconds

  ## function: reward
  ## tuned for correct safety vs efficiency trade off
  lambda = 0.0005
    
  ## function: nextState
  ## use minE, maxE, minN, maxN from Grid
  response = 0.5 # how often the pilot responds
  ## no action, turning probability distribution - must be 5 to items to match turns
  ## turns are adding -2*stepHeadingState, -1*stepHeadingState, 0, stepHeadingState, stepHeadingState
  turnDist = [0.05, 0.25, 0.4, 0.25, 0.05]

  #############################################################
  ##              Parameters needed for debris               ##
  #############################################################

  ## return
  CSLVProblem(minE, maxE, minN, maxN, stepHeadingState, timeThres, acGrid, lvStates, noAlert, 
              actionArray, debris, nStates, nActions, maintainCost, headingLimit, startDebris, 
              endDebris, timeStepSeconds, intersectTime, lambda, acSpeed, response, turnDist, 
              train, restricted, oldDebris, minErecord, maxErecord, minNrecord, maxNrecord,
              distanceDivision, distanceBuffer, timeStep, lvTime, safeThres)
end

end # module
