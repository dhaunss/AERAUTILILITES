import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import sys

from six.moves import input # python 2 and 3 compatible code

"""
Plot map of AERA. Should work for python 2 and 3 so far.
Call: python plot_aera_map.py outfilename_without_extension.
Code previously on wiki https://www.auger.unam.mx/AugerWiki/AERA_GPS_surveys by Ewa
Added to repo as wiki is not the best place for version managment etc.
Files for SD, AMIGA and AERA coordinates also copied from wiki.
"""

try:
    ofname = sys.argv[1]
except Exception as e:
    print("give a name for output file as argument")
    import sys
    sys.exit()

#options input--------------------------------------------------------------------------
print('This program plots a map of the AERA field. You can now define what you want to be plotted (antennas, beacon, CRS, FD Coihueco)\nDefault values are marked as _option_\n')
phasestypeIn = input('Do you want the deployment phases or the antenna type to be plotted? Enter "phases" or _"antennas"_:') or "antennas"
if not phasestypeIn in ['phases', 'antennas']: print('wrong input!')
DEandNL = input('Do you want the KIT/BUW and NIKHEF antennas to be plotted in different colors? Enter _"y"_ or "no":') or "y"
sparseRdIn = input('Do you want to highlight the sparse 1500m RD stations? Enter "y" or _"n"_:') or "n"
beaconIn = input('Do you want the beacon to be plotted? Enter "y" or _"n"_:') or "n"
CRSIn = input('Do you want the CRS to be plotted? Enter "y" or _"n"_:') or "n"
SDIn = input('Do you want the SD stations to be plotted? Enter _"y"_ or "n":') or "y"
FDIn = input('Do you want Coihueco to be plotted? Enter _"y"_ or "n":') or "y"
AMIGAIn = input('Do you want the AMIGA counters to be plotted? Enter "y" or _"n"_:') or "n"


#read in data---------------------------------------------------------------------------
aeraGPSfile = np.genfromtxt( 'AERA_coordinates_DB_3_all.txt', skip_header=4, usecols=[0,3,4], dtype=None, encoding=None)
if SDIn == 'y': SdGPSfile = np.genfromtxt( 'Sd_coordinates.txt', usecols=[1,2] )
if AMIGAIn == 'y': AMIGAGPSfile = np.genfromtxt( 'AMIGA_UC_coordinates.txt', usecols=[1,2] )

# sort for type of station--------------------------------------------------------------
# AERA different phases
AERAphaseI = [x for x in range(1,25)]
AERAphaseII = [ 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 91, 92, 93, 94, 98, 99, 100, 101, 102, 103, 107, 108, 109, 110, 111, 112, 113, 118, 119, 120, 121, 122, 123, 124, 125, 126, 131, 132, 133, 134, 135, 136, 137, 142, 143, 144, 145, 146, 151, 152, 157 ]
AERAphaseIII = [86, 89, 90, 95, 96, 97, 104, 106, 114, 115, 127, 128, 129, 130, 138, 139, 140, 141, 147, 148, 149, 150, 105, 116, 117 ]
# NIKHEF stations phase I, II
AERA_NIKHEF_I = [ ]#[ 2, 4, 5, 8, 9, 12 ]
AERA_NIKHEF_II = [ 34, 35, 36, 43, 44, 49, 58, 59, 60, 69, 70, 83, 84, 85, 88, 92, 93, 94, 100, 101, 102, 103, 110, 111, 112, 113, 122, 123, 124, 125, 126, 135, 136, 137, 144, 145, 146, 151, 152, 157 ]
# standard thinned out AERA grid
sparse_1500m_array = [118, 121, 71, 76, 82, 127, 129, 138, 148, 150]

mytype = 'S14, f, f'
AERAphaseIgps = np.array( [] , dtype=mytype )
AERAphaseIIgps = np.array( [] , dtype=mytype )
AERAphaseIIIgps = np.array( [] , dtype=mytype )
AERA_NIKHEF_I_gps = np.array( [] , dtype=mytype )
AERA_NIKHEF_II_gps = np.array( [] , dtype=mytype )
AERA_1500m_gps = np.array( [] , dtype=mytype )

for line in aeraGPSfile:
  if line[0].startswith('AERA'):
    AERAstation = int(line[0].split('_')[1])
    # different phases
    if AERAstation in AERAphaseI: AERAphaseIgps = np.append( AERAphaseIgps, np.array(line, dtype=mytype) )
    elif AERAstation in AERAphaseII: AERAphaseIIgps = np.append( AERAphaseIIgps, np.array(line, dtype=mytype) )
    elif AERAstation in AERAphaseIII: AERAphaseIIIgps = np.append( AERAphaseIIIgps, np.array(line, dtype=mytype) )
    else: print('something is wrong')
    # NIKHEF stations
    if AERAstation in AERA_NIKHEF_I: AERA_NIKHEF_I_gps = np.append( AERA_NIKHEF_I_gps, np.array(line, dtype=mytype) )
    elif AERAstation in AERA_NIKHEF_II: AERA_NIKHEF_II_gps = np.append( AERA_NIKHEF_II_gps, np.array(line, dtype=mytype) )
    # sparse RD grid
    if AERAstation in sparse_1500m_array: AERA_1500m_gps = np.append( AERA_1500m_gps, np.array(line, dtype=mytype) )
  # beacon, CRS
  elif line[0] == 'BEACON':
    beacon = [ line[1], line[2] ]
  elif line[0] == 'CRS-REF-ROOF':
    CRS = [ line[1], line[2] ]

#labels and colors------------------------------------------------------------------------
if phasestypeIn == 'phases': 
  labelAERAI = 'AERA phase I'
  labelAERAII ='AERA phase II'
  labelAERAIII ='AERA phase III'
elif phasestypeIn == 'antennas':
  labelAERAI = 'AERA LPDA'
  labelAERAII = 'AERA butterfly'
labelCRS = 'CRS'
labelBeacon = 'beacon'
labelSd = 'Water Cherenkov Detector'
labelAMIGA = 'AMIGA Unitary Cell'
labelFd = 'FD site'
labelSparse = '1500m Rd array'
colorAERAI = 'r'
colorAERAII = '#5A10FF'
colorAERAIII = 'seagreen'
colorCRS = 'sienna'
colorBeacon = 'sienna'
colorSd = color='0.7'#'#FF9A32'
colorAMIGA = 'b'
colorFd = 'lime'
mpl.rcParams.update({'font.size': 14})

#plot map---------------------------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111)

if SDIn == 'y': ax.plot( SdGPSfile.T[1], SdGPSfile.T[0], 'o', color=colorSd, markersize=5, markeredgecolor='None', label=labelSd )
ax.plot( AERAphaseIgps['f2'], AERAphaseIgps['f1'], '^', color=colorAERAI, markersize=4.5, markeredgecolor='k', label=labelAERAI )
if DEandNL == 'y': ax.plot( AERA_NIKHEF_I_gps['f2'], AERA_NIKHEF_I_gps['f1'], '^', color='1', markersize=4.5, markeredgecolor='k')
ax.plot( AERAphaseIIgps['f2'], AERAphaseIIgps['f1'], 'v', color=colorAERAII, markersize=4.5, markeredgecolor='k', label=labelAERAII )
if DEandNL == 'y': ax.plot( AERA_NIKHEF_II_gps['f2'], AERA_NIKHEF_II_gps['f1'], 'v', color='1', markersize=4.5, markeredgecolor='k')
if phasestypeIn == 'phases': ax.plot( AERAphaseIIIgps['f2'], AERAphaseIIIgps['f1'], '>', color=colorAERAIII, markersize=4.5, markeredgecolor='k', label=labelAERAIII)
elif phasestypeIn == 'antennas': ax.plot( AERAphaseIIIgps['f2'], AERAphaseIIIgps['f1'], 'v', color=colorAERAII, markersize=4.5, markeredgecolor='k')
if sparseRdIn == 'y': ax.plot( AERA_1500m_gps['f2'], AERA_1500m_gps['f1'], 'o', color='none', markersize=10, markeredgecolor='C1', label=labelSparse)
if AMIGAIn == 'y': ax.plot( AMIGAGPSfile.T[1], AMIGAGPSfile.T[0], 'o', color='k', markersize=6, markeredgecolor='k', label=labelAMIGA )
if CRSIn == 'y': ax.plot( CRS[1], CRS[0], 's', color=colorCRS, markersize=5, markeredgecolor='k', label=labelCRS )
if beaconIn == 'y': ax.plot( beacon[1], beacon[0], 'h', color=colorBeacon, markersize=8, markeredgecolor='k', label=labelBeacon )

#FD site
if FDIn == 'y':
  FD, = ax.plot( beacon[1], beacon[0], 'o', color=colorFd, markersize=12, markeredgecolor='k', label=labelFd, alpha = 0.5 ,markeredgewidth=2  )
  wedgeFD = mpatches.Wedge((445331.977,6114109.440), 1000, 240, 60, ec="none",facecolor="lime",alpha=0.3,label="FD FOV")
  ax.add_patch(wedgeFD)
  for ang in [-120,-90,-60,-30,0,30,60]:    
      ax.plot((445331.977,445331.977+np.cos(np.deg2rad(ang))*1000 ),(6114109.440,6114109.440+np.sin(np.deg2rad(ang))*1000),"k")

  # old Ewa code, HEAT down pos
  #ax.plot((445331.977,445331.977+np.cos(np.deg2rad(15))*700 ),(6114109.440,6114109.440+np.sin(np.deg2rad(15))*700),"k",linewidth=0.5)
  #ax.plot((445331.977,445331.977+np.cos(np.deg2rad(-15))*700 ),(6114109.440,6114109.440+np.sin(np.deg2rad(-15))*700),"k",linewidth=0.5)
  #HEAT1 = mpatches.Wedge((445331.977,6114109.440), 700, -15, 15, ec="none",facecolor='b',alpha=0.3,label="HEAT FOV")
  #HEAT2 = mpatches.Wedge((445331.977,6114109.440), 700, -60, -30, ec="none",facecolor='b',alpha=0.3)
  #HEAT3 = mpatches.Wedge((445331.977,6114109.440), 700, 30, 60, ec="none",facecolor='b',alpha=0.3)+

  # HEAT up pos
  ax.plot((445331.977,445331.977+np.cos(np.deg2rad(23))*700 ),(6114109.440,6114109.440+np.sin(np.deg2rad(23))*700),"k",linewidth=0.5)
  ax.plot((445331.977,445331.977+np.cos(np.deg2rad(70))*700 ),(6114109.440,6114109.440+np.sin(np.deg2rad(70))*700),"k",linewidth=0.5)
  ax.plot((445331.977,445331.977+np.cos(np.deg2rad(-20))*700 ),(6114109.440,6114109.440+np.sin(np.deg2rad(-20))*700),"k",linewidth=0.5)
  ax.plot((445331.977,445331.977+np.cos(np.deg2rad(-70))*700 ),(6114109.440,6114109.440+np.sin(np.deg2rad(-70))*700),"k",linewidth=0.5)
  HEAT1 = mpatches.Wedge((445331.977,6114109.440), 700, -30, 30, ec="none",facecolor='b',alpha=0.3,label="HEAT FOV")
  HEAT2 = mpatches.Wedge((445331.977,6114109.440), 700, -70, -30, ec="none",facecolor='b',alpha=0.3)
  HEAT3 = mpatches.Wedge((445331.977,6114109.440), 700, 30, 70, ec="none",facecolor='b',alpha=0.3)
  ax.add_patch(HEAT1)
  ax.add_patch(HEAT2)
  ax.add_patch(HEAT3)

#draw scale
ax.plot((448000,449000),(6115900,6115900),'k-', linewidth=2)
ax.text(448100, 6116000, '1 km')

#plot settings, labels, legend
ax.set_xlabel('east in km')
ax.set_ylabel('north in km')
ax.legend(loc=(0,0.8), numpoints=1, prop={'size':12}, frameon=True, ncol=2)
ax.axis('off')
ax.axis('equal')
ax.set_autoscale_on(False)
if FDIn == 'y' or beaconIn == 'y': 
  ax.set_xlim(443500,455000)
else:
  ax.set_xlim(447000,455000)
ax.set_ylim(6111000,6118000)

#save plot
for iend in ['.png','.pdf']:
  fig.savefig(ofname+iend, dpi=500)

fig.clf()
ax.cla()

