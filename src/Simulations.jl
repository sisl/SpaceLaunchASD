using Distributed

@everywhere begin

using NearestNeighbors
using GridInterpolations
using DelimitedFiles
using Statistics
using ProgressMeter
using SharedArrays
using Mmap

mutable struct historicInfo
  xEllipse::Array{Float64,1}
  yEllipse::Array{Float64,1}
  xFoci::Array{Float64,1}
  yFoci::Array{Float64,1}
end

mutable struct simulationInfo
  threshold::Int64
  flightNum::Int64
  weight::Array{Float64,1}
end

mutable struct flightInfo
  xArray::NTuple{101,Array{Float64,1}}
  yArray::NTuple{101,Array{Float64,1}}
  nameArray::Array{String,1}
end

struct necData
    timeEnd::Int64
    numCuts::Int64
    headingCutPoints::Array{Float64,1}
    hcpLength::Int64
    anomCutPoints::Array{Float64,}
    acpLength::Int64
end

function historicInfo()
  xEllipse = [32610.88497258722,30853.99592752435,29057.704761250094,27229.33503304424,25376.343013902104,23506.28617405547,21626.7912302343,19745.52190490307,17870.146551020538,16008.305796264774,14167.580360012316,12355.459194864128,10579.308101915387,8846.338965604977,7163.579749630955,5537.845390253171,3975.709717337977,2483.4785268145206,1067.1639208089123,-267.5399762868733,-1515.279826060004,-2671.0643179137182,-3730.283763122764,-4688.727969425025,-5542.602334315005,-6288.5421052050115,-6923.624764668075,-7445.380508886214,-7851.800797312318,-8141.344961155005,-8312.944867592018,-8366.007645636153,-8300.416488072515,-8116.529552030744,-7815.176988252454,-7397.656136188917,-6865.724928481132,-6221.593554217985,-5467.914435659689,-4607.770577742124,-3644.662353807057,-2582.4927945171826,-1425.5514499223755,-178.4968972175875,1153.6620312353057,2565.5862232976624,4051.627388199705,5605.850204688275,7222.055436563639,8893.803935313143,10614.441448928032,12377.124155542544,14174.844839930884,16000.459630456664,17846.71521339642,19706.2764408959,21571.754248044763,23435.733793566855,25290.802737645066,27129.57956920241,28944.741893729555,30729.054591402302,32475.39775388257,34176.79430683963,35826.4372238931,37417.71623651487,38944.243943324815,40399.88122143921,41778.76184195975,43075.31619148867,44284.294001756745,45400.7859901298,46420.244314903364,47338.50175106481,48151.78949452399,48856.75350581243,49450.46930788998,49930.455157049226,50294.683510916286,50541.59072330905,50670.0849020629,50679.55187308848,50569.859201532345,50341.358229298996,49994.88409700164,49531.75372773057,48953.76175985904,48263.17442618091,47462.721387133766,46555.5855364754,45545.39080854966,44436.18802706139,43232.4388460543,41938.997844384685,40561.092845408355,39104.303543697235,37574.538530288664,35978.01081726969,34321.2119711453,32610.88497258722]
  yEllipse = [-42015.98153097936,-42421.75880334291,-42623.74221286279,-42621.10240012086,-42413.83678976857,-42002.769815624226,-41389.549768805475,-40576.64227872794,-39567.320451082574,-38365.651701128685,-36976.4813347578,-35405.41294369066,-33658.78569483763,-31743.64860719278,-29667.731922570216,-27439.415689005556,-25067.695687571224,-22562.1468448131,-19932.884283692452,-17190.522176031445,-14346.130568726625,-11411.190364534312,-8397.546645889388,-5317.360537017996,-2183.0598054973225,992.7115906331392,4197.144800104517,7417.318006871828,10640.248774614889,13852.946591017611,17042.46539551829,20195.955875022162,23300.717313955665,26344.24878771806,29314.299492261383,32198.918007042663,34986.500293857214,37665.83624026132,40226.15456306555,42657.165895006656,44949.10388582663,47092.76415785339,49079.540965470114,50901.46141770683,52551.21713346866,54022.19320959485,55308.494392919485,56404.96835888342,57307.226010722276,58011.65872510339,58515.452481949214,58816.598828293725,58913.90263810055,58806.98664220726,58496.29271472147,57983.07991439459,57269.41929162918,56358.18548378413,55253.04513344257,53958.44217600689,52479.58005469212,50822.40093234199,48993.56198071178,47000.408838779746,44850.946342275754,42553.8066369653,40118.21479813949,37553.95208838532,34871.31699484315,32081.08419587529,29194.46161528362,26223.04572992782,23178.775303739687,20073.88372766525,16920.850151064962,13732.349595327922,10521.202245093256,7300.322116374256,4082.6653040241704,881.178013369582,-2291.2554175128776,-5421.864297634352,-8498.042957786953,-11507.40151365305,-14437.815812603065,-17277.476361917958,-20014.936039556946,-22639.156392758283,-25139.552334783006,-27506.035055941575,-29729.052971653844,-31799.630537734927,-33709.40477121732,-35450.65932392704,-37016.35596562714,-38400.16334373695,-39596.482897531234,-40600.47181613879,-41408.06294163991,-42015.98153097936]
  xFoci = [12152.975231141812,30169.987627709863]
  yFoci = [47656.58674620234,-31389.09003724606]
  return historicInfo(xEllipse, yEllipse, xFoci, yFoci)
end

function simulationInfo()
  ## 2 * focal length
  threshold = 102909
  flightNum = 10100
  # time of anomaly weight
  weight = [0.5,0.061069089,0.057339656,0.053837975,0.050550139,0.047463088,0.04456456,0.041843043,0.039287727,0.036888461,0.034635716,0.032520545]
  return simulationInfo(threshold, flightNum, weight)
end

function flightInfo()
  include("../assets/flightPaths.jl")
  xArray = xAll
  yArray = yAll
  nameArray = nameAll 
  return flightInfo(xArray, yArray, nameArray)
end

function necData()
  numCuts = 200 
  timeEnd = 81 + 1
  headingCutPoints = collect(-180.0:15.0:180.0)
  hcpLength = length(headingCutPoints)
  anomCutPoints = collect(-1.0:11.0)
  acpLength = length(anomCutPoints)
  necData(timeEnd, numCuts, headingCutPoints, hcpLength, anomCutPoints, acpLength)
end

hI = historicInfo()
sI = simulationInfo()
fI = flightInfo()
solveData = necData()

## when in radians
function directionSpeedsRad(acSpeed, head)
    xSpeed = round(acSpeed*cos(head))
    ySpeed = round(acSpeed*sin(head)) 
    return xSpeed, ySpeed
end

## when in degrees
function directionSpeedsDeg(acSpeed, head)
    xSpeed = round(acSpeed*cosd(head))
    ySpeed = round(acSpeed*sind(head)) 
    return xSpeed, ySpeed;
end

##           Useful Fucntions            ##
###########################################

## finds heading between two points
function angles(y1,y2,x1,x2)
    rad = atan((y1-y2), (x1-x2))
    phi = ParallelVI_.unwrap_rad(rad)
    return phi, rad2deg(phi)
end 

## calculates the distance of a list of points
function calcDistance(xList,yList)
    dist = 0
    for i = 1:(length(xList)-1)
        dist = dist + sqrt((xList[i]-xList[i+1])^2+(yList[i]-yList[i+1])^2)
    end
    return dist
end

## process debris
function formatDebris(profile::Int64)
    debrisFileName = string("../assets/debris",string(profile),".txt")
    ## GET and FORMAT debris data
    debris_raw = float(open(readdlm,debrisFileName))
    ## setup to be as big as the number of timesteps
    nTimesteps = 81       
    all_debris = [Any[] for _ in 1:nTimesteps]

    for i = 1:Base.size(debris_raw,1)
      push!(all_debris[Int(debris_raw[i])], debris_raw[i,2:end])
    end

    ## process to remove duplicates
    for i = 1:length(all_debris)
      if length(all_debris[i]) > 0
        for j = 1:length(all_debris[i])
          all_debris[i][j][4] = floor(floor(all_debris[i][j][4],digits=-1)/10.)
        end
      end
      all_debris[i] = unique(all_debris[i])
    end

    ## process into a dictionary
    debrisNow = Dict{Tuple{Int64,Int64},NearestNeighbors.KDTree}()

    for i = 1:length(all_debris)
      if length(all_debris[i]) > 0
        current_time_debris = hcat(all_debris[i]...)
        current_time_debris = current_time_debris[4,:],current_time_debris[1,:],current_time_debris[2,:]
        current_time_debris = hcat(current_time_debris...)
        current_time_debris = sort(current_time_debris, dims=1)
        # j = time debris passes through threshold
        for j = current_time_debris[1]:current_time_debris[Base.size(current_time_debris,1)]
          indexes = searchsorted(current_time_debris[:,1],j)
          if length(indexes) > 0
            data = current_time_debris[indexes,2:3]
            # key is time of anomaly, time debris passes through threshold
            debrisNow[(i,j)] = KDTree(Array(data'))
          end
        end
      end
    end
    return debrisNow
end

debris = formatDebris(2)

## calculates in the in and out point of a plane through historic boundary
function inAndOut(xList, yList, hI::historicInfo, sI::simulationInfo) # for historic
    inInd, outInd = 0, 0
    for i = 1:length(xList)
        if sqrt((xList[i]-hI.xFoci[1])^2+(yList[i]-hI.yFoci[1])^2) + sqrt((xList[i]-hI.xFoci[2])^2+(yList[i]-hI.yFoci[2])^2) < sI.threshold
            outInd = i
        end
    end
    for i = length(xList):-1:1
        if sqrt((xList[i]-hI.xFoci[1])^2+(yList[i]-hI.yFoci[1])^2) + sqrt((xList[i]-hI.xFoci[2])^2+(yList[i]-hI.yFoci[2])^2) < sI.threshold
            inInd = i
        end
    end
    return inInd, outInd
end

## prints simulation results
function printResults(sI::simulationInfo, dist, re, thru, distN)
  println()
  println("RESULTS:")
  finalDist = sum(dist.*sI.weight)
  finalRe = sum(re.*sI.weight)
  finalThru = sum(thru.*sI.weight)
  println("weighted number rerouted")
  println(round(finalRe,digits=2))
  println("% rerouted")
  println(round(finalRe*100/sI.flightNum,digits=2))
  println("average weighted distance")
  println(round(finalDist,digits=2))
  println("average added distance")
  println(round(finalDist-distN,digits=2))
  println("weighted number traverse 10x safety region")
  println(round(finalThru,digits=2))
  println("percent traverse 10x safety  region")
  println(round(finalThru*100/sI.flightNum,digits=2))
  println()
end

function actionSim(solveData::necData, point)
  timeRem = Int64(point[5])
  if isfile("../results/qvalues_grid_$(timeRem)")
    qvalue, gridX, gridY = open("../results/qvalues_grid_$(timeRem)", "r+") do f
      x = read(f, Int)
      y = read(f, Int)
      a = read(f, Int)
      b = read(f, Int)
      gridX = zeros(a)
      gridY = zeros(b)
      for i = 1:a
          gridX[i] = read(f, Float64)
      end
      for i = 1:b
          gridY[i] = read(f, Float64)
      end
      qvalue = Mmap.mmap(f, Matrix{Float64}, (x, y))
      qvalue, gridX, gridY
    end
    grid = RectangleGrid(gridX, gridY, solveData.headingCutPoints, solveData.anomCutPoints)

    ids, probs = interpolants(grid, point[1:4])
    actionList = sum(qvalue[ids,:].*probs,dims=1)
    return argmax(actionList)[2]
  else
    return 1
  end
end

## find debris ellipse and determine if aircraft within debris ellipse
function inCircle(problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State, cX::Float64, cY::Float64)
  ## check if aircraft is within debris ellipse
  curDist = sqrt((state.x-cX)^2+(state.y-cY)^2)
  if curDist <= problem.safeThres
    return 1., 1.
  else
    return 0., 0.
  end
end

## simulation distance reward 
function distanceRewardSim(debris::Dict{Tuple{Int64,Int64},NearestNeighbors.KDTree}, problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State)
  ## check if when launch vehicle passes through altitude
  if state.timeRem == problem.intersectTime
    distance = sqrt((state.x - problem.lvStates[6].x)^2+(state.y - problem.lvStates[6].y)^2)
    ## check distance between launch vehicle and aircraft
    if distance <= problem.lvStates[6].safe
      return -1., 1. # reward, ellipse
    end
  end
  ## if anomaly, determine if aircraft is at risk from any debris
  if state.anomaly >= 0.
    reward, ellipse = debrisLocationsSim(debris, problem, state)
    return -reward, ellipse
  end
  return 0., 0. # reward, ellipse -> not within safety threshold of debris
end

## find the locations of the debris
function debrisLocationsSim(debris::Dict{Tuple{Int64,Int64},NearestNeighbors.KDTree}, problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State)
  ## no active debris
  anomaly_time = Int(state.anomaly)
  ## setup how many closest points in the KDTree to collect
  debris_to_compare = 25
  ## only concerned with one debris profile at a time
  # look in dictionary only when time of anomaly is available
  if state.anomaly >= problem.startDebris && state.anomaly <= problem.endDebris
    ## for additional buffer dont just care about the time the debris passes through
    ## the threshold (center time) but also -20, -10, +10, and +20 seconds of that time
    center_time = problem.timeThres - state.timeRem
    for i = center_time-2:center_time+2
      ## for additional buffer also calculate if the anomaly occurs at the specified time
      ## as well as -10 and +10 seconds
      for j = anomaly_time-1:anomaly_time+1
        ## look if debris exists for this time of anomaly and current time
        if haskey(debris,(j,i))
          current_debris = get(debris,(j,i),0)
          if length(current_debris.data) < debris_to_compare
            idxs, dists = knn(current_debris, [state.x, state.y], length(current_debris.data))
          else
            idxs, dists = knn(current_debris, [state.x, state.y], debris_to_compare)
          end
          for j = 1:length(idxs)
            ## check if current position hits debris
            hitD, inE = inCircle(problem, state, current_debris.data[j][1], current_debris.data[j][2])
            if inE == 1.
              #println("here")
              return 1., 1. # returnValue, ellipseStatus
            end
          end ## end of set of debris check
        end ## end of debris if statement
      end ## end of anomaly time
    end ## end of time of anomaly
  end ## end of debris profile
  return 0., 0.
end

## setup the velocity reward that depends on heading change
function velocityRewardSim(problem::CSLVProblem_.CSLVProblem, action::CSLVProblem_.Action)
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

## set overall reward value
function rewardSim(debris::Dict{Tuple{Int64,Int64},NearestNeighbors.KDTree}, problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State, action::CSLVProblem_.Action)
  return problem.lambda * velocityRewardSim(problem, action) + distanceRewardSim(debris, problem, state)[1]
end

function nextStateSim(problem::CSLVProblem_.CSLVProblem, state::CSLVProblem_.State, action::CSLVProblem_.Action)
  ## if first time step empty
  if state.timeRem == 0.
    return ((0.,0.),(zeros(5),zeros(5)))
  end
  ## setup arrays for valid speeds and positions

  ## based on action, possible responses and headings
  if action.head != problem.noAlert
    ## does not respond, does respond
    response = [1-problem.response,problem.response]
    if problem.restricted
        headings = [ParallelVI_.unwrap_deg(state.head), ParallelVI_.unwrap_deg(state.head + action.head)]
    else
        headings = [ParallelVI_.unwrap_deg(state.head), action.head]
    end
  else
    ## turning probability distribution
    response = problem.turnDist
    headings = [ParallelVI_.unwrap_deg(state.head-2*problem.stepHeadingState), ParallelVI_.unwrap_deg(state.head-problem.stepHeadingState),
                ParallelVI_.unwrap_deg(state.head), ParallelVI_.unwrap_deg(state.head+problem.stepHeadingState),
                ParallelVI_.unwrap_deg(state.head+2*problem.stepHeadingState)]
  end
  xSpeeds = Array{Float64}(length(headings))
  ySpeeds = Array{Float64}(length(headings))
  possibleEast = Array{Float64}(length(headings))
  possibleNorth = Array{Float64}(length(headings))

  ## setup the speeds and positions
  for i = 1:length(headings)
    xSpeeds[i] = round(problem.acSpeed*cosd(headings[i]))
    ySpeeds[i] = round(problem.acSpeed*sind(headings[i]))
    possibleEast[i] = state.x + xSpeeds[i]
    possibleNorth[i] = state.y + ySpeeds[i]
    ## check within bounds and set to limits if outside
    if possibleEast[i] <= -2.5e4 || possibleEast[i] >= 5.1e4
      possibleEast[i] = state.x
    end
    if possibleNorth[i] <= -4.5e4 || possibleNorth[i] >= 6.5e4
      possibleNorth[i] = state.y
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
            probResponse[j] = response[1]*(1. - problem.lvStates[i].probAnom)
            probResponse[j+length(headings)] = response[2]*(problem.lvStates[i].probAnom)
          else
            nextStates[2*j-1] = nextStatesNoAnom[j]
            nextStates[2*j] = nextStatesAnom[j]
            probResponse[2*j-1] = response[j]*(1. - problem.lvStates[i].probAnom)
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

end

##          nominal simulation           ##
###########################################

function nominal(debris, problem, fI::flightInfo, sI::simulationInfo)
  ## setup final distance and thru 10X safety threshold metrics
  allDist, allThru = Float64[], Float64[]
  ## simulate!!
  @showprogress 1 "Computing..." for k = -1:10 ## potential times of anomaly
      sleep(0.1)
      ## for this time of anomaly setup distance and thru 10X safety threshold metrics
      anomDist, anomThru = Float64[], Float64[]
      for L = 1:length(fI.xArray) ## L denotes flight in array of flight data
          x, y = fI.xArray[L], fI.yArray[L] ## x and y lists from flight data
          numPositions = length(x)
          ## for this flight setup distance and thru 10X safety threshold metrics
          fightDist = Float64[]
          flightThru = 0.
          for j = 1:100 ## j denotes offset time 
              ## for this offset time setup thru 10X safety threshold metrics and lists of X and Y positions
              offsetThru = 0.
              offsetX, offsetY = Float64[], Float64[]
              push!(offsetX, x[1])
              push!(offsetY, y[1])
              position = 1
              for t = 1:1000 ## excess of possible time steps, terminates when no more positions to vist
                  timeRemaining = 81 - t + j
                  ## find current heading
                  phi, phiDeg = angles(y[position+1], offsetY[t], x[position+1], offsetX[t])
                  speeds = directionSpeedsRad(problem.acSpeed, phi)
                  ## update thru 10X safety threshold metric
                  if distanceRewardSim(debris, problem, CSLVProblem_.State(offsetX[t], offsetY[t], phiDeg, k, timeRemaining))[2] == 1.
                      offsetThru = offsetThru + 1
                  end
                  ## update X and Y position lists
                  push!(offsetX, offsetX[t]+speeds[1])
                  push!(offsetY, offsetY[t]+speeds[2])
                  ## find next way-point and see if past all available way-points
                  phiNew, phiDegNew = angles(offsetY[t+1], y[position+1], offsetX[t+1], x[position+1])
                  if sign(phi) == sign(phiNew)
                      position = position + 1 ## fixed?
                      if position + 1 > numPositions
                          break
                      end
                  end 
                end ## end of t, end of 1 flight path iteration (1 time of anomaly, 1 offset)
              distance = calcDistance(offsetX,offsetY) ## find distance of just completed flight path
              push!(fightDist, distance)
              ## update flight 10X safety threshold metric
              if offsetThru >= 1
                  flightThru = flightThru + 1
              end
            end ## end of j, end of 1 flight path iteration (1 time of anomaly, all offsets)
          ## update anomaly distance and 10X safety threshold metrics
          push!(anomDist, mean(fightDist))
          push!(anomThru, flightThru)
        end ## end of L, end of all flights (1 time of anomaly, all offsets)
      ## update all distance and 10X safety threshold metrics 
      push!(allDist, mean(anomDist))
      push!(allThru, sum(anomThru))
    end ## end of k, all flights (all times of anomalies, all offsets)
    ## print results
    printResults(sI, allDist, 0., allThru, sum(allDist.*sI.weight))
  ## return final distance for use in historic and proposed simulations
    return (round(sum(allDist.*sI.weight),digits=2))
end

##          historic simulation          ##
###########################################

function historic(debris, problem, fI::flightInfo, hI::historicInfo, sI::simulationInfo, distNom)
  ## setup final distance, number rerouted, and thru 10X safety threshold metrics
  allDist, allRe, allThru = Float64[], Float64[], Float64[]
  ## simulate!!
  @showprogress 1 "Computing..." for k = -1:10 ## potential times of anomaly
      sleep(0.1)
      ## for this time of anomaly setup distance, number rerouted, and thru 10X safety threshold metrics
      anomDist, anomRe, anomThru = Float64[], Float64[], Float64[]
      for L = 1:length(fI.xArray) ## L denotes flight in array of flight data
          x, y = fI.xArray[L], fI.yArray[L] ## x and y lists from flight data
          numPositions = length(x)
          ## for this flight setup distance , reroute, and thru 10X safety threshold metrics
          flightDist, offsetDist = Float64[], Float64[]
          flightRe, flightThru = 0., 0.
          for j = 1:100 ## j denotes offset time 
              ## for this offset time setup lists of X and Y positions
              offsetXnom, offsetYnom = Float64[], Float64[]
              push!(offsetXnom,x[1])
              push!(offsetYnom,y[1])
              position = 1
              ## for this offset time setup in and out locations, and lists of above and below X and Y positions
              inLoc, outLoc = 0., 0.
              offsetXa, offsetYa, offsetXb, offsetYb = Float64[], Float64[], Float64[], Float64[]
              extXell = cat(hI.xEllipse,hI.xEllipse,dims=1)
              extYell = cat(hI.yEllipse,hI.yEllipse,dims=1)
              ## for this offset time setup distance, and thru 10X safety threshold metrics
              offsetDist = Float64[]
              offsetThru = 0.
              for t = 1:1000 ## excess of possible time steps, terminates when no more positions to vist
                  ## must simulate nominal
                  timeRemaining = 81 - t + j
                  ## find current heading
                  phi, phiDeg = angles(y[position+1], offsetYnom[t], x[position+1], offsetXnom[t])
                  speeds = directionSpeedsRad(problem.acSpeed, phi)
                  ## update nominal X and Y position lists
                  push!(offsetXnom, offsetXnom[t]+speeds[1])
                  push!(offsetYnom, offsetYnom[t]+speeds[2])
                  ## find next way-point and see if past all available way-points
                  phiNew, phiDegNew = angles(offsetYnom[t+1], y[position+1], offsetXnom[t+1], x[position+1])
                  if sign(phi) == sign(phiNew)
                      position = position + 1 ## fixed?? 
                      if position + 1 > numPositions
                          break
                      end
                  end  
              end ## end of t, end of 1 nominal flight path iteration (1 time of anomaly, 1 offset)
              ## use nominal paths to find the historic paths (adjust around static ellipse)
              ## update in and out locations
              inInd, outInd = inAndOut(offsetXnom, offsetYnom, hI, sI)
              ## find when hits restricted region and move around it (check going above and below)
              inEllipseDist, outEllipseDist = Float64[], Float64[]
              if inInd != 0. && outInd != 0. 
                    flightRe = flightRe + 1
                  ## length until hit unsafe region
                  for i = 1:inInd
                      push!(offsetXa, offsetXnom[i])
                      push!(offsetYa, offsetYnom[i])
                      push!(offsetXb, offsetXnom[i])
                      push!(offsetYb, offsetYnom[i])
                  end
                  ## find region of safety region that needs to be avoided
                  for i = 1:length(hI.xEllipse)
                      inDist = sqrt((offsetXnom[inInd]-hI.xEllipse[i])^2 + (offsetYnom[inInd]-hI.yEllipse[i])^2)
                      outDist = sqrt((offsetXnom[outInd]-hI.xEllipse[i])^2 + (offsetYnom[outInd]-hI.yEllipse[i])^2)
                      push!(inEllipseDist, inDist)
                      push!(outEllipseDist, outDist)
                  end
                  ## move around the safety region until out of unsafe region
                  inLoc, outLoc = argmin(inEllipseDist), argmin(outEllipseDist)
                  if inLoc > outLoc
                      outLoc = outLoc + length(hI.xEllipse)
                  end
                  for i = inLoc:outLoc
                      push!(offsetXa, extXell[i])
                      push!(offsetYa, extYell[i])
                  end
                  for i = inLoc+length(hI.xEllipse):-1:outLoc
                      push!(offsetXb, extXell[i])
                      push!(offsetYb, extYell[i])
                  end
                  for i = outInd:length(offsetXnom)
                      push!(offsetXa, offsetXnom[i])
                      push!(offsetYa, offsetYnom[i])
                      push!(offsetXb, offsetXnom[i])                        
                      push!(offsetYb, offsetYnom[i])
                  end
              end
              ## find which rerouted distance is shorter and use that distance
              distanceA = calcDistance(offsetXa, offsetYa)
              distanceB = calcDistance(offsetXb, offsetYb)
              if distanceA <= distanceB ## update this ofset distance
                  push!(offsetDist, distanceA) 
                  for i = 1:(length(offsetXa)-1)
                      ## check if goes into an unsafe region
                      phi = atan((offsetYa[i+1]-offsetYa[i]),(offsetXa[i+1]-offsetXa[i])) ## updated

                      ## update thru 10X safety threshold metric
                      if distanceRewardSim(debris, problem, CSLVProblem_.State(offsetXa[i], offsetYa[i], rad2deg(phi), k, 0.))[2] == 1. 
                          offsetThru = offsetThru + 1
                      end
                  end
              else
                  push!(offsetDist, distanceB)
                  for i = 1:(length(offsetXb)-1)
                      ## check if it goes in an unsafe region
                      phi = atan((offsetYb[i+1]-offsetYb[i]),(offsetXb[i+1]-offsetXb[i])) ## updated
                      ## update thru 10X safety threshold metric
                      if distanceRewardSim(debris,problem, CSLVProblem_.State(offsetXb[i], offsetYb[i], rad2deg(phi), k, 0.))[2] == 1.
                          offsetThru = offsetThru + 1
                      end
                  end
              end
              if offsetThru >= 1
                  flightThru = flightThru + 1
              end
          end ## end of j, end of 1 flight path iteration (1 time of anomaly, all offsets)
          ## update anomaly distance and 10X safety threshold metrics
          push!(anomDist, mean(offsetDist))
          push!(anomRe, flightRe)
          push!(anomThru, flightThru)
      end ## end of L, end of all flights (1 time of anomaly, all offsets)
      ## update all distance and 10X safety threshold metrics 
      push!(allDist,mean(anomDist))
      push!(allRe,sum(anomRe))
      push!(allThru,sum(anomThru))
  end ## end of k, all flights (all times of anomalies, all offsets)
  ## print results
  printResults(sI, allDist, allRe, allThru, distNom)
end

##          proposed simulation          ##
###########################################

function proposed(debris, problem, fI::flightInfo, sI::simulationInfo, distNom)
  ## setup final distance, number rerouted, and thru 10X safety threshold metrics
  allDist, allRe, allThru = SharedArray{Float64}(12), SharedArray{Float64}(12), SharedArray{Float64}(12)
  ## simulate!!
  #@showprogress 1 "Computing..." 
  @sync @distributed for k = -1:10 ## potential times of anomaly
  #    sleep(0.1) 
      ## for this time of anomaly setup distance, number rerouted, and thru 10X safety threshold metrics
      anomDist, anomRe, anomThru = Float64[], Float64[], Float64[]
      ## L denotes which flight in array of flight data
      for L = 1.:length(fI.xArray) ## L denotes flight in array of flight data
          x, y = fI.xArray[L], fI.yArray[L] ## x and y lists from flight data
          numPositions = length(x)
          ## for this flight setup distance, number rerouted, and thru 10X safety threshold metrics
          flightDist = Float64[]
          flightRe, flightThru = 0., 0.
          ## j denotes offset time
          for j = 1.:100. ## j denotes offset time setup thru 10X safety threshold metrics and lists of X and Y positions
              ## for this offset time setup lists for X and Y
              offsetX, offsetY = Float64[], Float64[]
              push!(offsetX,x[1])
              push!(offsetY,y[1])
              ## for this offset time setup number rerouted and thru 10X safety threshold metrics
              offsetRe, offsetThru = 0., 0.
              position = 1
              for t = 1:1000 # excess of possible time steps, terminates when no more positions to vist
                  timeRemaining = 81 - t + j
                  ## find desired heading to next way-point
                  phiDes, phiDegDes = angles(y[position+1],offsetY[t], x[position+1], offsetX[t])
                  phi, phiDeg = phiDes, phiDegDes
                  #end ## update becuase i think you dont want to go arbitrary heading, you want to go desired
                  ## only have actions for state space
		              if timeRemaining <= 81 - k
                      act = actionSim(solveData, [offsetX[t], offsetY[t], phiDeg, k, timeRemaining])
                  else
                      act = actionSim(solveData, [offsetX[t], offsetY[t], phiDeg, -1, timeRemaining])
		  end
                  if act == 1 ## act == 1 means NIL and simulated to continue on same heading
                      if abs(phiDegDes - phiDeg)>30.1
                          if abs(phiDeg+30-phiDegDes) > abs(phiDeg-30-phiDegDes)
                              phiDeg = phiDeg-30
                              phi = phiDeg*pi/180
                          else
                              phiDeg = phiDeg+30
                              phi = phiDeg*pi/180
                          end
                      end
                      speeds = directionSpeedsRad(problem.acSpeed, phi)
                  else 
                      if problem.restricted
                          speeds = directionSpeedsDeg(problem.acSpeed, ParallelVI_.unwrap_deg(phiDeg+problem.actionArray[act].head))
                      else
                          speeds = directionSpeedsDeg(problem.acSpeed, problem.actionArray[act].head)
                      end
                  end
                  ## update rerouted metric
                  if act != 1
                      offsetRe = offsetRe + 1
                  end
                  ## update 10X safety threshold metric
                  if distanceRewardSim(debris, problem, CSLVProblem_.State(offsetX[t], offsetY[t], phiDeg, k, timeRemaining))[2] == 1.
                      offsetThru = offsetThru + 1
                  end
                  # update the next position to offsetX and offsetY
                  push!(offsetX, offsetX[t]+speeds[1])
                  push!(offsetY, offsetY[t]+speeds[2])
                  # find next way-point and see if past all available way-points
                  phiNew, phiDegNew = angles(offsetY[t+1], y[position+1], offsetX[t+1], x[position+1])
                  if sign(phi) == sign(phiNew)
                      position = position + 1 #### fixed?? 
                      if position + 1 > numPositions
                          break
                      end
                  end ## end of check
              end ## end of t, end of 1 flight path iteration (1 time of anomaly, 1 offset)
              ## update flight rerouted metric
              if offsetRe >= 1
                  flightRe = flightRe + 1
              end
              ## update flight 10X safety threshold metric
              if offsetThru >= 1
                  flightThru = flightThru + 1
              end
              ## update flight distance metric
              distance = calcDistance(offsetX, offsetY)
              push!(flightDist, distance)
          end ## end of j, end of 1 flight path iteration (1 time of anomaly, all offsets)
          ## update anomaly distance, number rerouted, and 10X safety threshold metrics
          push!(anomDist, mean(flightDist))
          push!(anomRe, flightRe)
          push!(anomThru, flightThru)  
      end ## end of L, end of all flights (1 time of anomaly, all offsets)
      ## update all distance, number rerouted, and 10X safety threshold metrics 
      allDist[k+2]=mean(anomDist)
      allRe[k+2]=sum(anomRe)
      allThru[k+2]=sum(anomThru)
  end ## end of k, all flights (all times of anomalies, all offsets)
  ## print results
  printResults(sI, allDist, allRe, allThru, distNom)
end
