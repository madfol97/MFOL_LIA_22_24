# Section to find pixels to include in gates
import numpy as np
import math
import datetime

def findNearestPixel(gateEdgeCoords, tlon, tlat):
    """
    Function to find nearest pixel in curvilinear grid to the gates edges
    
    Input:
    gateEdgeCoords ([lon_start, lat_start, lon_end, lat_end]): Coordinates of two edge points of the gate
    tlon (x,y): Longitudes in 2d
    tlat (x,y): Latitudes in 2d

    Returns:
    [indices_start:(x,y), indices_end:(x,y)]: Indices in 2d of the edge points.
    """
    distance_start = np.sqrt((tlon - gateEdgeCoords[0])**2 + (tlat - gateEdgeCoords[1])**2)
    min_indices_start = np.unravel_index(distance_start.argmin(axis=None), distance_start.shape)
    distance_end = np.sqrt((tlon - gateEdgeCoords[2])**2 + (tlat - gateEdgeCoords[3])**2)
    min_indices_end = np.unravel_index(distance_end.argmin(axis=None), distance_end.shape)
    return [min_indices_start, min_indices_end]

def findSlopeOfGate(strPt, endPt):
    """
    Function to find the gate slope
    Input:
    strPt (x,y): Dataset pixels of starting point in 2d
    endPt (x,y):  Dataset pixels of ending point in 2d

    Returns:
    a: slope
    """
    dx = endPt[0] - strPt[0]
    dy = endPt[1] - strPt[1]

    # Calculate slope (a)
    a = float(dy) / float(dx) # a = dy/dx
    return a

def findSlopeFromCoord(tlon1,tlat1,tlon2,tlat2):
    """
    Function to find the gate slope
    Input:
    strPt (x,y): Dataset pixels of starting point in 2d
    endPt (x,y):  Dataset pixels of ending point in 2d

    Returns:
    a: slope
    """
    dx = tlat1 - tlat2
    dy = tlon1 - tlon2

    # Calculate slope (a)
    a = float(dy) / float(dx) # a = dy/dx
    return a

def pixelTraversalAlgorithm(strPt, endPt, isVertical):
    """
    Function to select all pixels intercepting the line between two
    points. Digital Diff. Analyser (DDA) algorithm
    1) From starting and ending point, compute the line equation (slope and origin)
    2.1) From initial point, iterate in x with half a pixel width increments. Compute corresponding y with line equation. 
       Round (x,y) coordinates to select the corresponding pixel. Add pixel to list.
    OR
    2.2) From initial point, iterate in y with half a pixel height increments.  Compute corresponding x and add points.
    Seems to work for small distances. Would need to increment along coordinates instead of indexes for longer distances

    Input:
    strPt (x,y): Dataset pixels of starting point in 2d
    endPt (x,y):  Dataset pixels of ending point in 2d
    isVertical bool: If set to True, iterate in y instead of x. Initially, both iteration
        in x and y ensure to include all relevent pixels. For my purpose I needed only 
        one or the other since my gates were either vertical enough or horizontal enough to
        include all points. If iterate in both x and y, pixels will not be sorted.

    Returns:
    ptsIndexes: All indexes of intersepting points.
    """
    # Calculate direction vector
    dx = endPt[0] - strPt[0]
    dy = endPt[1] - strPt[1]

    # Determine step directions. Indicates whether to move + or - along axes
    stepX = 1 if dx > 0 else -1 if dx < 0 else 0
    stepY = 1 if dy > 0 else -1 if dy < 0 else 0

    # Calculate indices of starting and ending points (integers)
    sXIndex = int(strPt[0])
    sYIndex = int(strPt[1])
    eXIndex = int(endPt[0])
    eYIndex = int(endPt[1])

    ptsIndexes = [] # result
    x = sXIndex # Initialize first point
    y = sYIndex
    pt = (x, y)  # First point

    # If line vertical: iterate over y
    if dx == 0:
        for h in range(0, abs(dy) + 1):
            pt = (x, y + (stepY * h))
            ptsIndexes.append(pt)
        return ptsIndexes

    # Calculate slope (a) and interception point (b)
    a = float(dy) / float(dx) # a = dy/dx
    b = strPt[1] - strPt[0] * a # b = y - ax

    # Split the cell size in half and increment of half cell
    # size in both x and y to make sure line intersect at cell
    # boundaries
    sXIdxSp = round(2.0 * sXIndex) / 2.0
    sYIdxSp = round(2.0 * sYIndex) / 2.0
    eXIdxSp = round(2.0 * eXIndex) / 2.0
    eYIdxSp = round(2.0 * eYIndex) / 2.0

    prevPt = (np.nan, np.nan)
    if isVertical:
        # Iterate along y axis, half a grid size at a time
        # Calculate the corresponding X coordinate based on the line equation.
        # Add each pt
        for h in range(0, abs(dy) * 4):
            y = stepY * (h / 2.0) + sYIdxSp
            x = (y - b) / a
            if (stepY < 0 and y < eYIdxSp) or (stepY > 0 and y > eYIdxSp):
                break
            pt = (int(x), int(y))

            if prevPt != pt:
                ptsIndexes.append(pt)
                prevPt = pt
    else: 
        # Iterate along x axis, half a grid size at a time
        # Calculate the corresponding Y coordinate based on the line equation.
        # Add each pt
        for w in range(0, abs(dx) * 4):
            x = stepX * (w / 2.0) + sXIdxSp
            y = x * a + b
            # Break if out of ending bndry
            if (stepX < 0 and x < eXIdxSp) or (stepX > 0 and x > eXIdxSp):
                break

            pt = (int(x), int(y))

            if prevPt != pt:
                ptsIndexes.append(pt)
                prevPt = pt
    return ptsIndexes

def findPixelFromCoord(gateEdgeCoords, tlon, tlat):
    """
    Function to find nearest pixel in curvilinear grid to the gates points
    
    Input:
    gateEdgeCoords ([lon_start, lat_start, lon_end, lat_end]): Coordinates of two edge points of the gate
    tlon (x,y): Longitudes in 2d
    tlat (x,y): Latitudes in 2d

    Returns:
    [indices_start:(x,y), indices_end:(x,y)]: Indices in 2d of the gate points.
    """
    distance_start = np.sqrt((tlon - gateEdgeCoords[0])**2 + (tlat - gateEdgeCoords[1])**2)
    min_indices_start = np.unravel_index(distance_start.argmin(axis=None), distance_start.shape)
    distance_end = np.sqrt((tlon - gateEdgeCoords[2])**2 + (tlat - gateEdgeCoords[3])**2)
    min_indices_end = np.unravel_index(distance_end.argmin(axis=None), distance_end.shape)
    return [min_indices_start, min_indices_end]

def appendPointToArray(points, point_to_append):
    if point_to_append not in points:
        points.append(point_to_append)
    return points

def selectEdgesOfGate(indexesOfGate, isVertical):
    """
    Function to select the grid edges of a gate instead of centers of grid cells
    
    Input:
    indexesOfGate [(x1,y1),(x2,y2),...]: All indexes of intersepting points
    isVertical bool: If true, select point left-up corner

    Returns:
    [(x1,y1),(x2,y2),...]: Indices in 2d of the edge points.
    """
    uResult = []
    
    # First point down left corner
    uResult.append((indexesOfGate[0][0], indexesOfGate[0][1]-1))
    prev_lon = indexesOfGate[0][0]
    prev_lat = indexesOfGate[0][1]-1

    for i in range(len(indexesOfGate)):
        pnt_lon = indexesOfGate[i][0]
        pnt_lat = indexesOfGate[i][1]-1 
        if isVertical:
            if pnt_lon != prev_lon and pnt_lat != prev_lat:
                uResult = appendPointToArray(uResult, (pnt_lon -1, pnt_lat))
            uResult = appendPointToArray(uResult, (pnt_lon, pnt_lat))
                
        else:
            # If previous is diagonal down or to the right, add lower left corner
            if pnt_lon != prev_lon:
                uResult = appendPointToArray(uResult, (pnt_lon, pnt_lat))

            # If last point, add bottom both corner
            if i == len(indexesOfGate) - 1: 
                if prev_lat != pnt_lat and prev_lon != pnt_lon:
                    uResult = appendPointToArray(uResult, (pnt_lon, pnt_lat))
                uResult = appendPointToArray(uResult, (pnt_lon-1, pnt_lat))

            else :
                next_lat = indexesOfGate[i+1][1]-1
                # If next is in diagonal or over the pixel, add lower right corner
                if next_lat != pnt_lat:
                    factor = -1
                    if pnt_lon - prev_lon > 0 :
                        factor = 1
                    uResult = appendPointToArray(uResult, (pnt_lon+factor, pnt_lat))

        prev_lon = pnt_lon
        prev_lat = pnt_lat
    return uResult

def selectVelocitiesCgridORCA1(uGate, gateName, uVel, vVel, aice, ulon, ulat):
    """
    Select the right C grid points velocities and ice concentration for ORCA 1 degree
    resolution. Pole is over Northern Canada under the CAA and grid turns around the pole 
    There's a fold in y = 290
    
    Input:
    uGate [x,y]: Gates U points pixels 
    gateName string: name of the gate
    uVel [x,y]: Dataset ice velocity in x in 2d
    vVel [x,y]: Dataset ice velocity in y in 2d
    aice [x,y]: Sea ice area (concentration) for each grid cells

    Returns:
    interVelocity []: velocity at center of gate segment (m/s),
    interaIce []: ice concentration at center of gate segment (0 to 1),
    dx []: lenght of gate segment (m)
    """
    interVelocity = []
    dx = []
    interaIce = []

    gatesUm1 = ['Nares']
    if gateName == 'QEI':
        uFactor = [1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1]
    elif gateName in gatesUm1: 
        uFactor = [-1 for i in range(len(uGate)-1)]
    else: 
        uFactor = [1 for i in range(len(uGate)-1)]
    for index in range(len(uGate)-1):
        # Segment from pt to pt+1
        segment = (uGate[index][0] - uGate[index + 1][0], uGate[index][1] - uGate[index + 1][1])
        desired_values = [(0, -1), (0, 1), (1, 0), (-1,0)]
        # if segment not in desired_values: 
        #     print('GATE GAP NOT VALID')
        #     print(segment)
        #     print(uGate[index][0], uGate[index][1])
        #     print(uGate[index + 1][0], uGate[index + 1][1])
            #return np.nan, np.nan, np.nan
        # Same longitude, vertical gate, v component of velocity through it
        if segment[0] == 0:
            step = 1  # Going along x
            if (segment[1] == 1): # Going against x
                step = 0
            pixel = uGate[index+step]
            # Already a C grid
            interVelocity.append(vVel[pixel[0], pixel[1]]* uFactor[index])

            # Interpolate dx
            segmentWidth = getMeasureInMeters(ulat[uGate[index][0], uGate[index][1]],ulon[uGate[index][0], uGate[index][1]], ulat[uGate[index+1][0], uGate[index+1][1]],ulon[uGate[index+1][0], uGate[index+1][1]])
            dx.append(segmentWidth)

            # Interpolate aice
            if pixel[0]==290: # Along the bipolar fold
                if pixel[1]==123:
                    pixel2ForIceInterpolation=(290,236)
                if pixel[1]==122:
                    pixel2ForIceInterpolation=(290,237)
            else: 
                pixel2ForIceInterpolation = (pixel[0]+1, pixel[1]) # + 1 along y 
            ratio = 0.5 # segment_a / (segment_a + segment_b)
            delta_aice = aice[pixel2ForIceInterpolation[0], pixel2ForIceInterpolation[1]] - aice[pixel[0], pixel[1]]
            aIceInterp = aice[pixel[0], pixel[1]] + (ratio * delta_aice)
            interaIce.append(aIceInterp)

        # Same latitude, horizontal gate
        elif segment[1] == 0:
            step=1 # going along y
            if (segment[0] == 1): # Going against y
                step = 0
            pixel = uGate[index+step]

            # Interpolate  u : which is vertical over pole
            # Already a C grid
            interVelocity.append(uVel[pixel[0], pixel[1]]*uFactor[index])

            # Interpolate dx
            segmentWidth = getMeasureInMeters(ulat[uGate[index][0], uGate[index][1]],ulon[uGate[index][0], uGate[index][1]], ulat[uGate[index+1][0], uGate[index+1][1]],ulon[uGate[index+1][0], uGate[index+1][1]])
            dx.append(segmentWidth)

            # Interpolate aice
            pixel2ForIceInterpolation = (pixel[0], pixel[1]+1) # + 1 along x
            ratio = 0.5 # segment_a / (segment_a + segment_b)
            delta_aice = aice[pixel2ForIceInterpolation[0], pixel2ForIceInterpolation[1]] - aice[pixel[0], pixel[1]]
            aIceInterp = aice[pixel[0], pixel[1]] + (ratio * delta_aice)
            interaIce.append(aIceInterp)
    return interVelocity, interaIce, dx

def selectVelocitiesBgrid_ne120_t12(uGate, gateName, uVel, vVel, aice, htn,hte):
    """
    Select the right B grid points velocities and ice concentration for CESM tripolar grid
    ne.120_t12. Pole is over Northern Canada under the CAA and grid turns around the pole 
    There's a fold in lon_index = 2399. Use for cesm1.3 hr
    
    Input:
    uGate [x,y]: Gates U points pixels 
    gateName string: name of the gate
    uVel [x,y]: Dataset ice velocity in x in 2d
    vVel [x,y]: Dataset ice velocity in y in 2d
    aice [x,y]: Sea ice area (concentration) for each grid cells

    Returns:
    interVelocity []: velocity at center of gate segment (m/s),
    interaIce []: ice concentration at center of gate segment (0 to 1),
    dx []: lenght of gate segment (m)
    """
    interVelocity = []
    dx = []
    interaIce = []
    
    vFactor = 1
    uFactor = 1
    # Factors depends on the grid and the positive/negative flux definition
    gatesmV1 = ['Pr. Gustaf Adolf', 'Peary','Sverdrup','Nares', 'Nansen Sound','Eureka Sound','Fram'] 
    gatesmUI = [ 'Fitzwilliam','Kellett Strait', 'Crozier Channel', 'Pr. Gustaf Adolf', 'Peary','Sverdrup','Nares', 'Nansen Sound','Eureka Sound', 'Jones','Lancaster','Fram']
    if gateName in gatesmV1:
        vFactor = -1
    if gateName in gatesmUI: 
        uFactor = -1

    for index in range(len(uGate)-1):
        # Segment from pt to pt+1
        segment = (uGate[index][0] - uGate[index + 1][0], uGate[index][1] - uGate[index + 1][1])
        desired_values = [(0, -1), (0, 1), (1, 0), (-1,0)]
        if segment not in desired_values: 
            print('GATE GAP NOT VALID')
            print(segment)
            print(uGate[index][0], uGate[index][1])
            print(uGate[index + 1][0], uGate[index + 1][1])
            return np.nan, np.nan, np.nan
        
        # Vertical gate
        if segment[0] == 0:

            step = None
            if (segment[1] == -1): # Lat increasing (actual lat < next lat))
                step = 0
            elif segment[1] == 1: # Lat decreasing
                step = 1

            # i and j indices
            pixel_ij = (uGate[index+step][0], uGate[index+step][1])
            pixel_im1j = (pixel_ij[0], pixel_ij[1]-1)
            pixel_ip1j = (pixel_ij[0], pixel_ij[1]+1)
            pixel_ijp1 = (pixel_ij[0]+1, pixel_ij[1])
            if pixel_ijp1[0] > 2399:
                pixel_ijp1 = (pixel_ij[0], pixel_ij[1])

            # Interpolate 
            htn_ij = htn[pixel_ij].values
            htn_ip1j = htn[pixel_ip1j].values
            htn_im1j = htn[pixel_im1j].values

            dyv_im1j = (htn_im1j + htn_ij)/2
            dyv_ij = (htn_ij + htn_ip1j)/2

            v_ij_north = 0.5*(dyv_ij*(vVel[pixel_ij]/100)*vFactor + dyv_im1j*(vVel[pixel_im1j]/100)*vFactor)/htn_ij
            interVelocity.append(v_ij_north)
            
            dx.append(htn[pixel_ij].values)

            # Interpolate aice
            aIceInterp = (aice[pixel_ij] + aice[pixel_ijp1])/2
            interaIce.append(aIceInterp/100)

        # Horizontal gate
        elif segment[1] == 0:
            step = None
            if (segment[0] == -1): # Lon increasing (actual Lon < next Lon))
                step = 1
            elif segment[0] == 1: # Lon decreasing
                step = 0

            # i and j indices
            pixel_ij = (uGate[index+step][0], uGate[index+step][1]-1)
            pixel_ijm1 = (pixel_ij[0]-1, pixel_ij[1])
            pixel_ijp1 = (pixel_ij[0]+1, pixel_ij[1])
            pixel_ip1j = (pixel_ij[0], pixel_ij[1]+1)
            if pixel_ijp1[0] > 2399:
                pixel_ijp1 = (pixel_ij[0], pixel_ij[1])

            # Interpolate 
            hte_ij = hte[pixel_ij].values
            hte_ijm1 = hte[pixel_ijm1].values
            hte_ijp1 = hte[pixel_ijp1].values

            dyu_ijm1 = (hte_ijm1 + hte_ij)/2
            dyu_ij = (hte_ij + hte_ijp1)/2

            u_ij_east = 0.5*(dyu_ij*(uVel[pixel_ij]/100)*uFactor + dyu_ijm1*(uVel[pixel_ijm1]/100)*uFactor)/hte_ij
            interVelocity.append(u_ij_east)
            
            dx.append(hte[pixel_ij].values)
            
            # Interpolate aice
            aIceInterp = (aice[pixel_ij] + aice[pixel_ip1j])/2
            interaIce.append(aIceInterp/100)
        else: print('Error : Gate is neither vertical or horizontal!')
    return interVelocity, interaIce, dx

# RIOPS u, v coordinates have an impact. 
def interpolateBGrid(uGate, gateName, uVel, vVel, aice, e1t, e2t, e1v, e2u):
    interVelocity = []
    dx = []
    interaIce = []
    vFactor = -1
    uFactor = -1

    gatesV1 = ['Jones', 'Lancaster', 'Cardigan', 'Hell', 'Wellington', 'Penny', 'McDougall', 'Byam Martin Channel', 'Byam Channel', 'Austin Channel']
    gatesUI = [ 'Fitzwilliam','Kellett Strait', 'Crozier Channel', 'Cardigan', 'Hell', 'Wellington', 'Penny', 'McDougall', 'Byam Martin Channel', 'Byam Channel', 'Austin Channel']
    if gateName in gatesV1:
        vFactor = 1
    if gateName in gatesUI: 
        uFactor = 1
    for index in range(len(uGate)-1):
        # Segment from pt to pt+1
        segment = (uGate[index][0] - uGate[index + 1][0], uGate[index][1] - uGate[index + 1][1])
        desired_values = [(0, -1), (0, 1), (1, 0), (-1,0)]
        if segment not in desired_values: 
            print('GATE GAP NOT VALID')
            print(segment)
            print(uGate[index][0], uGate[index][1])
            print(uGate[index + 1][0], uGate[index + 1][1])
            return np.nan, np.nan, np.nan
        # Vertical gate 
        if segment[0] == 0:
            step = 1  #Going up
            if (segment[1] == 1): # Going down
                step = 0
            pixel_a = uGate[index+step]
            pixel_b = (pixel_a[0]+1, pixel_a[1]) # pixel_b is at the left of pixel_a
            segment_a = e2t[pixel_a[0], pixel_a[1]].values/2
            segment_b = e2t[pixel_b[0], pixel_b[1]].values/2

            # Interpolate 
            delta_v = vVel[pixel_b[0], pixel_b[1]] - vVel[pixel_a[0], pixel_a[1]]
            ratio = segment_a / (segment_a + segment_b)
            vInterp = vVel[pixel_a[0], pixel_a[1]] + (ratio * delta_v)
            interVelocity.append(vInterp* vFactor)

            # Interpolate dx
            dx.append(e1v[pixel_a[0], pixel_a[1]].values)

            # Interpolate aice
            delta_aice = aice[pixel_b[0], pixel_b[1]] - aice[pixel_a[0], pixel_a[1]]
            aIceInterp = aice[pixel_a[0], pixel_a[1]] + (ratio * delta_aice)
            interaIce.append(aIceInterp)

        # Same latitude, horizontal gate
        elif segment[1] == 0:
            step=0 # going towards right
            if (segment[0] == -1): # Going towards left
                step = 1
            pixel_a = uGate[index + step]
            pixel_b = (pixel_a[0], pixel_a[1]+1) # pixel_b is over pixel_a
            segment_a = e1t[pixel_a[0], pixel_a[1]].values/2
            segment_b = e1t[pixel_b[0], pixel_b[1]].values/2

            # Interpolate  u : which is vertical over pole
            delta_u = uVel[pixel_b[0], pixel_b[1]] - uVel[pixel_a[0], pixel_a[1]]
            ratio = segment_a / (segment_a + segment_b)
            uInterp = uVel[pixel_a[0], pixel_a[1]] + (ratio * delta_u)
            interVelocity.append(uInterp * uFactor)

            # Interpolate dx
            dx.append(e2u[pixel_a[0], pixel_a[1]].values)

            # Interpolate aice
            delta_aice = aice[pixel_b[0], pixel_b[1]] - aice[pixel_a[0], pixel_a[1]]
            aIceInterp = aice[pixel_a[0], pixel_a[1]] + (ratio * delta_aice)
            interaIce.append(aIceInterp)
    return interVelocity, interaIce, dx
    

# Works for RIOPS grid
def selectDx(gates,aice, e1v, e2u):
    """
    Function to compute segments of gates for B grid
    Input:
    gates [x,y]: array<{'name':string, 'shortName': string, 'coord: <array<number>>}> : [
    {'name': "Amundsen", 'shortName': 'Adm', 'coord' : [360-127.7, 70.5, 360-125.27, 72.17], 'slopeFactor': 0}]
    aice [x,y]: sea ice concentration
    e1v [x,y]: Measures of grid cell centered on V pint, along x axis
    e2u [x,y]: Measures of grid cell centered on U point, along y axis

    Returns: 
    dx []: lenght of gate segment (m)
    """
    # Gates definition
    gateLenght = {}
    pixelsOfGate = {}
    for gate in gates:
        dxi = 0.0
        dx = []
        pixels = []
        uGate = gate['gate']
        for index in range(len(uGate)-1):
            # Segment from pt to pt+1
            segment = (uGate[index][0] - uGate[index + 1][0], uGate[index][1] - uGate[index + 1][1])
            desired_values = [(0, -1), (0, 1), (1, 0), (-1,0)]
            if segment not in desired_values: 
                print('GATE GAP NOT VALID')
                print(segment)
                print(uGate[index][0], uGate[index][1])
                print(uGate[index + 1][0], uGate[index + 1][1])
                return np.nan, np.nan, np.nan
            # Same longitude, vertical gate 
            if segment[0] == 0:
                step = 1  #Going up
                if (segment[1] == 1): # Going down
                    step = 0
                pixel_a = uGate[index+step]
                pixel_b = (pixel_a[0]+1, pixel_a[1]) 
                # Interpolate dx
                if ~np.isnan(aice[pixel_a[0], pixel_a[1]]) and ~np.isnan(aice[pixel_b[0], pixel_b[1]]):
                    dx.append(e1v[pixel_a[0], pixel_a[1]].values)
                    pixels.append(pixel_a)

            # Same latitude, horizontal gate
            elif segment[1] == 0:
                step=0 # going towards right
                if (segment[0] == -1): # Going towards left
                    step = 1
                pixel_a = uGate[index + step]
                pixel_b = (pixel_a[0], pixel_a[1]+1) 

                # Interpolate dx
                if ~np.isnan(aice[pixel_a[0], pixel_a[1]]) and ~np.isnan(aice[pixel_b[0], pixel_b[1]]):
                    dx.append(e2u[pixel_a[0], pixel_a[1]].values)
                    pixels.append(pixel_a)

        for i in range(len(dx)):
            dxi += dx[i]
        gateLenght[gate['name']] = dxi
        pixelsOfGate[gate['name']] = pixels
        print('Gate ' + str(gate['name']) + ' - gate lenght : ' + str(dxi) + ' m' + str(range(len(dx))) + 'segments vs '+ str(index))
    
    return gateLenght, pixelsOfGate

def interpolategrid(uGate, gateName, uVel, vVel, aice, gridNature,  measures):
    """
    Function to compute ui and vi in middle of grid cell side
    by linear interpolation.
    Input:
    uGate [x,y]: Gates U points pixels 
    gateName string: name of the gate
    uVel [x,y]: Dataset ice velocity in x in 2d
    vVel [x,y]: Dataset ice velocity in y in 2d
    aice [x,y]: Sea ice area (concentration) for each grid cells
    gridNature string: "A", "B" or "C"
    measures : required for specific interpolation { 
        e1t [x,y]: Measures of grid cell centered on T point, along x axis
        e2t [x,y]: Measures of grid cell centered on T point, along y axis
        e1v [x,y]: Measures of grid cell centered on V pint, along x axis
        e2u [x,y]: Measures of grid cell centered on U point, along y axis
        }: Needed measures depending on nature of grid. Here it's a B grid

    Returns: 
    interVelocity []: velocity at center of gate segment (m/s),
    interaIce []: ice concentration at center of gate segment (0 to 1),
    dx []: lenght of gate segment (m)
    """
    if gridNature == "B":
        interVelocity, interaIce, dx = interpolateBGrid(uGate, gateName, uVel, vVel, aice, measures['e1t'], measures['e2t'], measures['e1v'], measures['e2u'])
    if gridNature == "C":
        interVelocity, interaIce, dx = selectVelocitiesCgridORCA1(uGate, gateName, uVel, vVel, aice, measures['ulon'], measures['ulat'])
    if gridNature == "B2":
        interVelocity, interaIce, dx = selectVelocitiesBgrid_ne120_t12(uGate, gateName, uVel, vVel, aice, measures['htn'], measures['hte'])

    return interVelocity, interaIce, dx


def computeIceFluxesCAA(tlon, tlat, uVel, vVel, iceConc, gridNature, measures): #, gates):
    """
    Function to compute the ice fluxes through all gates of the CAA
    Input:
    tlon [x,y]: Dataset grid longitudes in 2d
    tlat [x,y]: Dataset grid latitudes in 2d
    uVel [x,y]: Dataset ice velocity in x in 2d
    vVel [x,y]: Dataset ice velocity in y  in 2d
    iceConc [x,y]: Sea ice area (concentration) for each grid cells
    gridNature string: "A", "B" or "C"
    measures { 
        e1t [x,y]: Measures of grid cell centered on T point, along x axis
        e2t [x,y]: Measures of grid cell centered on T point, along y axis
        e1v [x,y]: Measures of grid cell centered on V pint, along x axis
        e2u [x,y]: Measures of grid cell centered on U point, along y axis
        }: Needed measures depending on nature of grid. Here it's a B grid


    Returns:
    gates [{
        'name':string,
        'shortName':'string',
        'coord': [lon1, lat1, lon2, lat2],
        'slopeFactor': double (Slopefactor to account for the grid (0,0) and orient the normal vector inward of the CAA),
        'iceFlux': {'total':double, 'pixels':[]}
        }]: Gates all relevent variables to compute ice fluxes.
        The ice flux through the gate is gate['iceFlux']['total]
        Resulting positive flux is an import into the CAA, negative is an export out of the CAA
    """
    # Gates definition
    gates = [
    {'name': "Amundsen", 'shortName': 'Adm', 'coord' : [360-127.7, 70.5, 360-125.27, 72.17], 'slopeFactor': 0},
    {'name': "M'Clure", 'shortName': 'Mclure','coord' : [236.0, 74.4, 360-122.2, 76.02], 'slopeFactor': 0}, #-124W = 236, -122.4 = 237.6
    {'name': "Ballantyne", 'shortName': 'Ball','coord' : [360-116.47, 77.56, 360-115.06, 77.95], 'slopeFactor': 0},
    {'name': "Wilkins", 'shortName': 'Wilk','coord' : [360-114.3, 78.08, 360-113.10, 78.3], 'slopeFactor': 0},
    {'name': "Pr. Gustaf Adolf", 'shortName': 'Gust','coord' : [360-110.4, 78.75, 360-105.5, 79.2], 'slopeFactor': 0},
    {'name': "Peary", 'shortName': 'Peary','coord' : [360-103.78, 79.35, 360-100.0, 79.9], 'slopeFactor': 0},
    {'name': "Sverdrup", 'shortName': 'Sv','coord' : [360-99.05, 80.10, 360-96.3, 80.3], 'slopeFactor': 0},
    {'name': "Nares", 'shortName': 'Nares','coord' : [360-61.36, 82.24, 360-59.9, 82.02], 'slopeFactor': np.pi},
    {'name': "Jones", 'shortName': 'Jones','coord' : [360-81.5, 75.78, 360-80.9, 76.2], 'slopeFactor': np.pi},
    {'name': "Lancaster", 'shortName': 'Lanc','coord' : [360-81.9, 73.7, 360-81.89, 74.52], 'slopeFactor': 0},
    {'name': "Cardigan", 'shortName': 'Card','coord' : [360-90.55, 76.6, 360-90.21, 76.55], 'slopeFactor': 0},
    {'name': "Hell", 'shortName': 'Hell','coord' : [360-89.84, 76.56, 360-89.5, 76.55], 'slopeFactor': 0},
    {'name': "Penny", 'shortName': 'Penny','coord' : [360-97.73, 76.5, 360-96.4, 76.67], 'slopeFactor': 0},
    {'name': "Wellington", 'shortName': 'Well','coord' : [360-93.5, 74.87, 360-91.95, 74.88], 'slopeFactor': 0},
    {'name': "McDougall", 'shortName': 'McDoug','coord' : [360-97.66, 75.16, 360-96.5, 75.07], 'slopeFactor': 0},
    {'name': "Byam Martin Channel", 'shortName': 'BMC','coord' : [360-105.68, 75.80, 360-103.65, 75.92], 'slopeFactor': 0},
    {'name': "Byam Channel", 'shortName': 'Byam','coord' : [360-105.93, 75.2, 360-104.68, 75.21], 'slopeFactor': 0},
    {'name': "Austin Channel", 'shortName': 'Austin','coord' : [360-103.93, 75.39, 360-102.52, 75.59], 'slopeFactor': 0},
    {'name': "Kellett Strait", 'shortName': 'Kellett','coord' : [360-118.6, 75.6, 360-117.46, 75.3], 'slopeFactor': np.pi},
    {'name': "Crozier Channel", 'shortName': 'Crozr','coord' : [360-119.89, 75.9, 360-119.07, 75.7], 'slopeFactor': np.pi},
    {'name': 'Fitzwilliam', 'shortName': 'Fitz', 'coord': [360-116.1, 76.66, 360-115.38, 76.44], 'slopeFactor': np.pi},
    {'name': "Nansen Sound", 'shortName': 'Nansn','coord' : [360-93.19, 81.3, 360-91.0, 81.60], 'slopeFactor': np.pi},
    {'name': "Eureka Sound", 'shortName': 'Eurka','coord' : [360-88.84, 78.2, 360-87.32, 78.17], 'slopeFactor': np.pi},
    ]

    for gate in gates:
        fluxGate = 0.0
        # Find gates pixels on grid
        gateEdgesPixels = findPixelFromCoord(gate['coord'], tlon, tlat)
        if gate['name'] == 'Jones' or gate['name'] == 'Lancaster':
            gatePixels = pixelTraversalAlgorithm(gateEdgesPixels[0], gateEdgesPixels[1], True)
            gate['gate'] = selectEdgesOfGate(gatePixels, True)
        else:
            gatePixels = pixelTraversalAlgorithm(gateEdgesPixels[0], gateEdgesPixels[1], False)
            gate['gate'] = selectEdgesOfGate(gatePixels, False)
        interVelocity, interaIce, dx= interpolategrid(gate['gate'], gate['name'], uVel, vVel, iceConc, gridNature, measures)

        for i in range(len(interVelocity)):
            ci = interaIce[i]
            ui = interVelocity[i]
            dxi = dx[i]

            # Flux sum(ci*ui*dxi)
            pixelFlux = ci*ui*dxi
            print(pixelFlux)
            if not math.isnan(pixelFlux):
                fluxGate += pixelFlux

        gate['flux'] = fluxGate
        print('Gate ' + str(gate['name']) + ' - Flux : ' + str(fluxGate) + ' m^2/s')

    return gates

def computeGatesOnce(gates, tlon, tlat):
    """
    Function to compute the ice fluxes through all gates of the CAA
    Input:
    gates: array<{'name':string, 'shortName': string, 'coord: <array<number>>}> : [
    {'name': "Amundsen", 'shortName': 'Adm', 'coord' : [360-127.7, 70.5, 360-125.27, 72.17], 'slopeFactor': 0}]

    tlon [x,y]: Dataset grid longitudes in 2d
    tlat [x,y]: Dataset grid latitudes in 2d

    Returns:
    gates [{
        'name':string,
        'shortName':'string',
        'coord': [lon1, lat1, lon2, lat2],
        'slopeFactor': double (Slopefactor to account for the grid (0,0) and orient the normal vector inward of the CAA),
        }]: Gates all relevent variables to compute ice fluxes.
        The ice flux through the gate is gate['iceFlux']['total]
        Resulting positive flux is an import into the CAA, negative is an export out of the CAA
    """
    # Gates definition
    for gate in gates:
        # Find gates pixels on grid
        gateEdgesPixels = findPixelFromCoord(gate['coord'], tlon, tlat)
        if gate['name'] == 'Jones' or gate['name'] == 'Lancaster':
            gatePixels = pixelTraversalAlgorithm(gateEdgesPixels[0], gateEdgesPixels[1], True)
            gate['gate'] = selectEdgesOfGate(gatePixels, True)
        else:
            gatePixels = pixelTraversalAlgorithm(gateEdgesPixels[0], gateEdgesPixels[1], False)
            gate['gate'] = selectEdgesOfGate(gatePixels, False)
    return gates

def computeIceFluxesCAAFromGates(gatesGrids, uVel, vVel, iceConc, gridNature, measures):
    """
    Function to compute the ice fluxes through all gates of the CAA
    Input:
    gates [{
        'name':string,
        'shortName':'string',
        'coord': [lon1, lat1, lon2, lat2],
        'slopeFactor': double (Slopefactor to account for the grid (0,0) and orient the normal vector inward of the CAA),
        'iceFlux': {'total':double, 'pixels':[]}
        }]: Gates all relevent variables to compute ice fluxes.
    uVel [x,y]: Dataset ice velocity in x in 2d
    vVel [x,y]: Dataset ice velocity in y  in 2d
    iceConc [x,y]: Sea ice area (concentration) for each grid cells 
    gridNature string: "A", "B" or "C"
    measures { 
        e1t [x,y]: Measures of grid cell centered on T point, along x axis
        e2t [x,y]: Measures of grid cell centered on T point, along y axis
        e1v [x,y]: Measures of grid cell centered on V pint, along x axis
        e2u [x,y]: Measures of grid cell centered on U point, along y axis
        }: Needed measures depending on nature of grid. Here it's a B grid


    Returns:
    fluxGates {}: Dict with all fluxes per gate name
        The ice flux through the gate is gate['iceFlux']['total]
        Resulting positive flux is an import into the CAA, negative is an export out of the CAA
    """
    fluxGates = {}
    for gate in gatesGrids:
        fluxGate = 0.0
        interVelocity, interaIce, dx = interpolategrid(gate['gate'], gate['name'], uVel, vVel, iceConc, gridNature, measures)
        gateNumberOfMeasures = 0
        for i in range(len(interVelocity)):
            ci = interaIce[i]
            ui = interVelocity[i]
            dxi = dx[i]

            # Flux sum(ci*ui*dxi)
            pixelFlux = ci*ui*dxi
            if not math.isnan(pixelFlux):
                fluxGate += pixelFlux
                gateNumberOfMeasures += 1

        fluxGates[gate['name']] = fluxGate
        # print('Gate ' + str(gate['name']) + ' - Flux : ' + str(fluxGate) + ' m^2/s')
        print('Gate ' + str(gate['name']) + ' - NumberOfMeasures : ' + str(gateNumberOfMeasures))

    return fluxGates

# Works for CESM1.3
def computeGatesLenghtInModel(gatesGrids, measures):
    """
    Function to compute the gate lenghts in the model based on the model measurements
    Input:
    gates [{
        'name':string,
        'shortName':'string',
        'coord': [lon1, lat1, lon2, lat2],
        'slopeFactor': double (Slopefactor to account for the grid (0,0) and orient the normal vector inward of the CAA),
        'iceFlux': {'total':double, 'pixels':[]}
        }]: Gates all relevent variables to compute ice fluxes.
    measures { 
        e1t [x,y]: Measures of grid cell centered on T point, along x axis
        e2t [x,y]: Measures of grid cell centered on T point, along y axis
        e1v [x,y]: Measures of grid cell centered on V pint, along x axis
        e2u [x,y]: Measures of grid cell centered on U point, along y axis
        }: Needed measures depending on nature of grid. Here it's a B grid


    Returns:
    gateLenghts {'dx': dx, 'nature': nature}: Dict with all gates dx and nature per gate name
    """
    gateLenghts = {}
    for gate in gatesGrids:
        uGate = gate['gate']
        dx = []
        nature = []
        gatelenght = 0.
        for index in range(len(uGate)-1):
            # Segment from pt to pt+1
            segment = (uGate[index][0] - uGate[index + 1][0], uGate[index][1] - uGate[index + 1][1])
            desired_values = [(0, -1), (0, 1), (1, 0), (-1,0)]
            if segment not in desired_values: 
                print('GATE GAP NOT VALID')
                print(segment)
                print(uGate[index][0], uGate[index][1])
                print(uGate[index + 1][0], uGate[index + 1][1])
                return np.nan, np.nan, np.nan
            
            # Vertical gate
            if segment[0] == 0:

                step = None
                if (segment[1] == -1): # Lat increasing (actual lat < next lat))
                    step = 0
                elif segment[1] == 1: # Lat decreasing
                    step = 1

                # i and j indices
                pixel_ij = (uGate[index+step][0], uGate[index+step][1])
                dx.append(measures['htn'][pixel_ij].values)
                nature.append('v')

            # Horizontal gate
            elif segment[1] == 0:
                step = None
                if (segment[0] == -1): # Lon increasing (actual Lon < next Lon))
                    step = 1
                elif segment[0] == 1: # Lon decreasing
                    step = 0

                # i and j indices
                pixel_ij = (uGate[index+step][0], uGate[index+step][1]-1)                
                dx.append(measures['hte'][pixel_ij].values)
                nature.append('h')
            else: print('Error : Gate is neither vertical or horizontal!')
        # Compute pythagore gate lenght
        tempVertSum = 0
        tempHorizSum = 0
        for i,elem in enumerate(dx):
            if nature[i] == 'h':
                tempHorizSum += elem
            elif nature[i] == 'v':
                tempVertSum += elem
        gatelenght = np.sqrt(tempHorizSum**2 + tempVertSum**2)
        gateLenghts[gate['name']] = {'dx': dx, 'nature': nature, 'gatelength':gatelenght}
        print('Gate ' + str(gate['name']) + ' - lenght : ' + str(gatelenght) + ' m')

    return gateLenghts


# DESUETS
# Compute date for number of days after 2004-09-25. Each year last 365 days
def calculateDateSinceRefDate(number_of_days, refDatetime):
    years, remainder = divmod(number_of_days, 365)  # Calculate the number of years and remaining days
    delta = datetime.timedelta(days=remainder)  # Create a timedelta object with the remaining days
    final_date = refDatetime + datetime.timedelta(days=365 * years) + delta  # Calculate the final date

    return final_date


def getMeasureInMeters(lat1, lon1, lat2, lon2):
    R = 6371229#6358.243 # Radius at 74N
    dLat = math.radians(lat2) - math.radians(lat1)
    dLon = math.radians(lon2) - math.radians(lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d # meters

def findGateNormal(a, slopeFactor):
    """
    Function to find the normal vector to the gate, pointing in the CAA
    Input:
    a : slope of the gate
    slopeFactor :  Factor to substract so that the normal vector points inward the CAA

    Returns:
    n: normal vector
    """
    if abs(a) != 0.0:
        n_a = -1/a # slope is reciprocal
    else: 
        n_a = 0.0
    theta= np.arctan(n_a) - slopeFactor
    n = np.array([np.cos(theta), np.sin(theta)])
    return n

