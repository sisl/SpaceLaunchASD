using Distributed

N_PROCS = 2

addprocs(N_PROCS-1)  

@everywhere using JLD
@everywhere using GridInterpolations
@everywhere using SharedArrays
@everywhere using Mmap
@everywhere __PARALLEL__ = true  

push!(LOAD_PATH, "../src")

@everywhere include("CSLVProblem_.jl")
@everywhere include("ParallelVI_.jl")
@everywhere using GridInterpolations
debris = load("../assets/debris.jld", "debris")

cores = 2

@everywhere struct necData
    timeEnd::Int64
    numCuts::Int64
    headingCutPoints::Array{Float64,1}
    hcpLength::Int64
    anomCutPoints::Array{Float64,}
    acpLength::Int64
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

solveData = necData()

# returns an array of active timesteps
function findActionTimesteps(problem::ParallelVI_.CSLVProblem_.CSLVProblem)
    actionTimesteps = findall(x -> x != problem.minErecord[1], problem.minErecord)
    if length(actionTimesteps) >= 1
        return vcat(actionTimesteps[1]-1, actionTimesteps)
    else 
        return actionTimesteps
    end
end

# This function save the qvalues and grid
@everywhere function saveMasterGridAndQtable(qvalue::Array{Float64,2}, grid::GridInterpolations.RectangleGrid, timeRem::Int64)
    open("../results/qvalues_grid_$(timeRem)", "w+") do f
        println("saving")
        write(f, size(qvalue,1))
        write(f, size(qvalue,2))
        write(f, length(grid.cutPoints[1]))
        write(f, length(grid.cutPoints[2]))
        write(f, grid.cutPoints[1])
        write(f, grid.cutPoints[2])
        write(f, qvalue)
    end
end

@everywhere function saveMasterGridAndQtable(qvalue::SharedArray{Float64,2}, grid::GridInterpolations.RectangleGrid, timeRem::Int64)
    open("../results/qvalues_grid_$(timeRem)", "w+") do f
        println("saving")
        write(f, size(qvalue,1))
        write(f, size(qvalue,2))
        write(f, length(grid.cutPoints[1]))
        write(f, length(grid.cutPoints[2]))
        write(f, grid.cutPoints[1])
        write(f, grid.cutPoints[2])
        write(f, qvalue)
    end
end

# reads in a current Master Q table and grid
# returns Master Q table and grid
@everywhere function openMasterQTable(timeRem::Int64, solveData::necData) 
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
    return qvalue, grid
end

# use the min and max of the grids to set the new grid area
# return the updated x and y grid values
@everywhere function setUpdatedGrid(grid1::GridInterpolations.RectangleGrid, grid2::GridInterpolations.RectangleGrid, solveData::necData)
    xMin = min(grid1.cutPoints[1][1], grid2.cutPoints[1][1])
    xMax = max(grid1.cutPoints[1][end], grid2.cutPoints[1][end])
    yMin = min(grid1.cutPoints[2][1], grid2.cutPoints[2][1])
    yMax = max(grid1.cutPoints[2][end], grid2.cutPoints[2][end])
    xGrid = collect(range(xMin, stop=xMax, length=solveData.numCuts))
    yGrid = collect(range(yMin, stop=yMax, length=solveData.numCuts))
    return xGrid, yGrid
end

# pull only this time from the grid and qvalues
@everywhere function timeGridAndQvalues(grid::GridInterpolations.RectangleGrid, qvalues::SharedArray{Float64,2}, timeRem::Int64, solveData::necData)
    xCuts, yCuts = grid.cutPoints[1], grid.cutPoints[2]
    grid = RectangleGrid(xCuts, yCuts, solveData.headingCutPoints, solveData.anomCutPoints)
    timeSize = length(xCuts)*length(yCuts)*solveData.hcpLength*solveData.acpLength
    qValues = qvalues[timeRem*timeSize+1:(timeRem+1)*timeSize,:]
    return grid, qValues
end

# check to make sure point is withiqvaluesn grid
@everywhere function checkPointLocation(point::Array{Float64,1}, grid::GridInterpolations.RectangleGrid)
    if point[1] < grid.cutPoints[1][end] && point[1] > grid.cutPoints[1][1] 
        if point[2] < grid.cutPoints[2][end] && point[2] > grid.cutPoints[2][1]
            return true
        end
    end
    return false
end

@everywhere function calcNewQvalues(grid::GridInterpolations.RectangleGrid, g1::GridInterpolations.RectangleGrid, q1::Array{Float64,2}, g2::GridInterpolations.RectangleGrid, q2::Array{Float64,2})
    q = Array{Float64}(undef, (length(grid), 6))
    for i = 1:length(grid)
        point = ind2x(grid,i)
        if checkPointLocation(point, g1)
            if checkPointLocation(point, g2)
                # update with g1 and g2
                ids, probs = interpolants(g1, point)
                a1 = sum(q1[ids,:].*probs, dims=1)
                ids, probs = interpolants(g2, point)
                a2 = sum(q2[ids,:].*probs, dims=1)
                q[i,:] = min.(a1, a2)
            else
                # update with just g1
                ids, probs = interpolants(g1, point)
                q[i,:] = sum(q1[ids,:].*probs, dims=1)
            end
        elseif checkPointLocation(point, g2)
            # update with just g2
            ids, probs = interpolants(g2, point)
            q[i,:] = sum(q2[ids,:].*probs, dims=1)
        else
        	q[i,:] = zeros(6)
        end
    end
    return q
end

@everywhere function incorporateDebris(problem::ParallelVI_.CSLVProblem_.CSLVProblem, curQvalues::SharedArray{Float64,2}, timeRem::Int64, solveData::necData)
    curGrid, curqvalues = timeGridAndQvalues(problem.acGrid[timeRem], curQvalues, timeRem, solveData)
    if isfile("../results/qvalues_grid_$(timeRem)")	
        curMasterQvalues, curMasterGrid = openMasterQTable(timeRem, solveData)
        newXgrid, newYgrid = setUpdatedGrid(curGrid, curMasterGrid, solveData)
        # setup new grid
        grid = RectangleGrid(newXgrid, newYgrid, solveData.headingCutPoints, solveData.anomCutPoints)
        # solve for new qvalues
        qvalue = calcNewQvalues(grid, curGrid, curqvalues, curMasterGrid, curMasterQvalues)
        saveMasterGridAndQtable(qvalue, grid, timeRem)
    else
        saveMasterGridAndQtable(curQvalues, curGrid, timeRem)
    end
end

function runLaunchVehicle(solveData::necData)
  for ta = -1.:11.
    println("On time of anomaly = $(ta)")
    for pt = 0.
        utility, qvalues, problem = ParallelVI_.parallelTest(cores, 108., 44., ta, pt) # use launch vehicle location 
        finalize(utility)
        timesteps = findActionTimesteps(problem)     
        if length(timesteps) >= 1
            @sync @distributed for timeRem in timesteps
                if timeRem < solveData.timeEnd
                    incorporateDebris(problem, qvalues, round(Int64,timeRem), solveData)
                end
            end
        else
            println("wont save")
        end   
        finalize(qvalues)
    end
  end

end


function runFullDebrisProfile(debris::Dict{Tuple{Int64,Int64},Array{Array,1}}, solveData::necData)
    # run all debris
    keyNum = 1
    for (key, points) in debris
        ta, pt = Float64(key[1]), Float64(key[2]) # time of anomaly, pass through time
        runKeysDebris(keyNum, key, points, ta, pt, solveData)
        keyNum = keyNum + 1
    end
    # run launch vehicle
    println("debris is done, calculating launch vehicle")
    runLaunchVehicle(solveData)
end

# Done
function runKeysDebris(keyNum::Int64, key, points, ta::Float64, pt::Float64, solveData::necData)
    l = length(points)
    for i = 1:l
        println("On key = $(key)")
        println("$(i)/$(l) pieces of debris for this key")
        x = round(points[i][1])
        y = round(points[i][2])
        utility, qvalues, problem = ParallelVI_.parallelTest(cores, x, y, ta, pt) 
        finalize(utility)
	    timesteps = findActionTimesteps(problem)     
        if length(timesteps) >= 1
       	    @sync @distributed for timeRem in timesteps
       	        if timeRem < solveData.timeEnd
             	    incorporateDebris(problem, qvalues, round(Int64,timeRem), solveData)
      	        end
            end
        else
       	    println("wont save")
        end   
        finalize(qvalues)
    end
end

runFullDebrisProfile(debris, solveData)

