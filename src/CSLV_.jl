module CSLV_

## to setup grid
using GridInterpolations

# ## define problem
include("CSLVProblem_.jl")

import Base.convert

## export Problem, State, Action
export CSLVProblem, State, Action
## export helping functions
export approxEq, unwrap_rad, unwrap_deg
## export reward functions
export velocityReward, stateAtAnomaly, statePostAnomaly, inEllipse, debrisLocations, 
       distanceReward, reward
## export next state function
export nextState
## export post processing functions
export value, action

## use this function to establish approximate equality
function approxEq(x,y,epsilon = 1.e-2)
  abs(x-y) < epsilon
end

## use this function to wrap angles when in radians
function unwrap_rad(a::Float64)
  if (a > pi)
      return mod(a + pi, 2*pi) - pi
  elseif (a < -pi)
      return -(mod(-a + pi, 2*pi) - pi)
  else
      return a
  end
end

## use this function to wrap angles when in degrees
function unwrap_deg(a_deg::Float64)
  unwrap_rad(a_deg*pi/180)*180/pi
end

## setup the velocity reward that depends on heading change
function velocityReward(problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State, action::CSLVProblem_.Action)
  ## no alert
  if action.head == problem.noAlert
    return 0.
  elseif abs(action.head) == problem.stepHeadingState
    return -0.5
  elseif abs(action.head) == 2*problem.stepHeadingState
    return -1.
  else
    return problem.maintainCost ## reward if alert commands current heading
  end
end

## find the launch vehicle state when anomaly occurs
function stateAtAnomaly(problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State)
  problem.lvStates[convert(Int,floor(state.anomaly))]
end

## find the launch vehicle state at the time step after anomaly if no anomaly had occurred
function statePostAnomaly(problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State)
  problem.lvStates[convert(Int,floor(state.anomaly + 1))]
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

## find the locations of the debris 
function debrisLocations(problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State)
  ## current location
  point = [state.x,state.y]
  # if state.anomaly >= 0.
  if state.anomaly == problem.debris.anomalyTime - 1 || state.anomaly == problem.debris.anomalyTime || state.anomaly == problem.debris.anomalyTime + 1
    if state.timeRem == problem.debris.remainingTime - 2 || state.timeRem == problem.debris.remainingTime - 1 || state.timeRem == problem.debris.remainingTime || state.timeRem == problem.debris.remainingTime + 1 || state.timeRem == problem.debris.remainingTime + 2
      hitD, inE = inEllipse(problem, state, problem.debris.eLocation, problem.debris.nLocation)
      if inE == 1.
        return 1., 1. # returnValue, ellipseStatus
      else
        return 0., 0. # returnValue, ellipseStatus
      end
    end
#    end
  end
  return 0., 0. # no active debris
end

## set the distance reward value
function distanceReward(problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State)
  ## not within safety threshold of debris
  reward = 0.
  ellipse = 0.
  ## if anomaly, determine if aircraft is at risk from any debris
  if state.anomaly >= 0.
    reward, ellipse = debrisLocations(problem, state)
    reward = -reward
  end
  reward, ellipse
end

## set overall reward value
function reward(problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State, action::CSLVProblem_.Action)
  if velocityReward(problem, state, action) == -Inf
    return -Inf
  else
    ## use lambda to weight safety vs. efficiency
    return problem.lambda * velocityReward(problem, state, action) + distanceReward(problem, state)[1]
  end
end

## find the potential next states
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
  xSpeeds = Array{Float64}(length(headings))
  ySpeeds = Array{Float64}(length(headings))
  possibleEast = Array{Float64}(length(headings))
  possibleNorth = Array{Float64}(length(headings))

  ## setup the speeds and positions
  for i = 1:length(headings)
    xSpeeds[i] = round(problem.acSpeed*cosd(headings[i]), 0)
    ySpeeds[i] = round(problem.acSpeed*sind(headings[i]), 0)
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
    nextStates = Array{Array}(length(headings))
    for i=1:length(headings)
      nextStates[i] = [possibleEast[i], possibleNorth[i], headings[i], state.anomaly, state.timeRem-1.]
    end
    return ((response),(nextStates))
  ## next state if anomaly has not occurred
  else
    nextStatesNoAnom = Array{Array}(length(headings))
    nextStatesAnom = Array{Array}(length(headings))
    for i = 1:length(problem.lvStates)
      ## check if anomaly can occur and setup potential next states
      if approxEq(problem.lvStates[i].timeLV, (state.timeRem-problem.timeThres))
        for j = 1:length(headings)
          nextStatesNoAnom[j] = [possibleEast[j], possibleNorth[j], headings[j], -1, state.timeRem-1.]
          nextStatesAnom[j] = [possibleEast[j], possibleNorth[j], headings[j], problem.lvStates[i].timeLV, state.timeRem-1.]
        end
        nextStates = Array{Array}(length(headings)*2)
        probResponse = Array{Float64}(length(headings)*2)
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

## during post processing, used to find the utility at a given states
function value(problem::CSLVProblem_.CSLVProblem, util, x)
  interpolate(problem.acGrid[round(Int,x[end])], util, x)
end

function value(grid::RectangleGrid, util, x)
  interpolate(grid, util, x)
end

function action(grid::RectangleGrid, act, x)
  id, prob = interpolants(grid,x)
  mostProbActPoint = ind2x(grid,id[indmax(prob)])
  return round(interpolate(grid,act,mostProbActPoint),0)
end

function value(problems::Array{CSLVProblem_.CSLVProblem,1}, utils::Array{Array{Float64,1},1}, x)
  values = [] 
  for i in 1:length(problems)
    push!(values,interpolate(problems[i].acGrid[round(Int,x[end])], utils[i], x))
  end
  return minimum(values)
end

## during post processing, used to find the optimal action at a given state
function action(problem::CSLVProblem_.CSLVProblem, util, x)
  ## want optimal, so start with min and find max
  QHi = 0.
  ind = 1 
  x[3] = unwrap_deg(x[3])
  ## cycle through actions
  for i in 1:problem.nActions
    ## find current utility value
    QNow = reward(problem, CSLVProblem_.State(x), problem.actionArray[i])
    ## find the next states and their probabilities of occurring
    (probabilities, nextStates) = nextState(problem, CSLVProblem_.State(x), problem.actionArray[i])
    ## cycle over all potential next states
    for nextStateIndex in 1:length(nextStates)
      xStar = nextStates[nextStateIndex] 
      if xStar[end] > 0
        QNow = QNow + probabilities[nextStateIndex] * GridInterpolations.interpolate(problem.acGrid[round(Int64,xStar[end])], util, xStar)
      end
    end
    ## set maximum utility value and record corresponding action
    if i == 1 || QNow > QHi
      QHi = QNow
      ind = i
    end
  end
  return ind
end

end ## module
