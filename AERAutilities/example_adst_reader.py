#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import ROOT
import os
import sys
import imp
import matplotlib.pyplot as plt





AugerOfflineRoot = os.environ["AUGEROFFLINEROOT"]
ROOT.gSystem.Load(os.path.join(AugerOfflineRoot, "lib/libRecEventKG.so"))

path_to_enum = '/home/dhaunss/software/AERAutilities/convertenums.py'

if not os.path.exists(path_to_enum):
	sys.exit("Path to enum file not correct, cant handle enums! Aborting ...")


ce = imp.load_source('', path_to_enum)
rdstQ = ce.rdstQ
rdshQ = ce.rdshQ

argv = sys.argv
if len(argv) <= 1:
	sys.exit("Aborting: Pass a ADST file as argument ...")

fname = argv[1]

def load_adst_energy_fluence(input_path):
	DataFiles = [fname]  # paths of ADST files
	print("processing files: ")
	print(DataFiles)
	identifier = []
	files = ROOT.std.vector('string')()
	for datfile in DataFiles:
		files.push_back(datfile)
		identifier.append(os.path.basename(datfile))
	
	file = ROOT.RecEventFile(files)
	event = ROOT.RecEvent()
	file.SetBuffers(event)
	
	geo = ROOT.DetectorGeometry()
	file.ReadDetectorGeometry(geo)
	
	# loop over all events in ADST files
	e_list = []
	while file.ReadNextEvent() == ROOT.RecEventFile.eSuccess:
		
		rEvent = event.GetRdEvent()
		rdstation = rEvent.GetRdStationVector()
		event_id  = rEvent.GetRdEventId()	
		#rShower = rEvent.GetRdRecShower()
		#sEvent = event.GetSDEvent()
		#sShower = sEvent.GetSdRecShower()
		
		for station in rdstation:
			if not station.HasParameter(rdstQ.eSignalEnergyFluenceMag):
				continue
			print(station.GetId(),station.GetParameter(rdstQ.eSignalEnergyFluenceMag))	
			e_list.push_back(station.GetParameter(rdstQ.eSignalEnergyFluenceMag))
			
	return e_fluences ={f"{event_id}": e_list}
	
def fast_plot(trace):
	np.save("true", trace)
	
	plt.hist((trace), bins=40)
	plt.yscale("log")
	plt.savefig("energydenoised.png")
	plt.show()
	