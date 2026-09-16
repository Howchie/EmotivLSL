#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.2.3),
    on September 16, 2026, at 12:25
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.2.3'
expName = 'nBack'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920,1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, data.getDateStr(format="%d-%m-%Y-%H%M"))
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\howch\\OneDrive\\Documents\\EEG Timing Tests\\DRT_timing_test.py',
        savePickle=False, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('debug')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('debug')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='Dell', color=(-1.0000, -1.0000, -1.0000), colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='pix',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = (-1.0000, -1.0000, -1.0000)
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'pix'
    if expInfo is not None:
        expInfo['frameRate'] = 240
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Intro" ---
    # Run 'Begin Experiment' code from code_6
    win.mouseVisible=False
    import serial
    import string
    import numpy as np
    from psychopy import visual, monitors
    import datetime
    import random
    import scipy.stats as ss
    import statistics
    import time
    from pylsl import StreamInfo, StreamOutlet, local_clock
    
    # Create marker stream for Lab Streaming Layer
    info = StreamInfo(name='PsychoPy Markers', type='Markers', channel_count=1,nominal_srate=0, 
                      channel_format='string', source_id=expInfo['participant'])
    outlet = StreamOutlet(info)  # Broadcast the stream.
    
    # Set experiment variables
    Resolution = (win.monitor.getSizePix())
    msg='doh!'#if this comes up we forgot to update the msg!
    FrameRate = 60
    DRT_Instance=0
    duration=120 # in seconds
    drt_running=False
    Stimulus="LED"
    
    # Functions required for timing
    def reset_drt_epoch(ser, timeout_s=1.0):
    	ser.reset_input_buffer();
    	old_timeout = ser.timeout;
    	ser.timeout = 0.01;
    
    	try:
    		t_send_ns = time.perf_counter_ns();
    		ser.write(b"resettimer\n");
    		ser.flush();
    
    		deadline_ns = t_send_ns + int(timeout_s * 1_000_000_000);
    
    		while time.perf_counter_ns() < deadline_ns:
    			raw = ser.readline();
    
    			# Capture immediately after the serial read returns.
    			t_recv_ns = time.perf_counter_ns();
    			t_recv_lsl = local_clock();
    
    			if not raw:
    				continue
    
    			line = raw.decode("utf-8", "replace").rstrip("\r\n");
    
    			if line.startswith("timer_reset>"):
    				board_tick = int(line.split(">", 1)[1].strip());
    
    				# Estimated PC perf_counter time when the board reset occurred.
    				anchor_pc_ns = (t_send_ns + t_recv_ns) // 2;
    
    				# Same estimated instant expressed in the LSL clock.
    				anchor_lsl = (
    					t_recv_lsl
    					- (t_recv_ns - anchor_pc_ns) / 1_000_000_000
    				);
    
    				return (
    					anchor_pc_ns,
    					anchor_lsl,
    					board_tick,
    					t_recv_ns - t_send_ns,
    				);
    
    	finally:
    		ser.timeout = old_timeout;
    
    	raise RuntimeError("Timed out waiting for timer_reset acknowledgement");
    
    DRT_PREFIX = "dta>DRT, SFT,";
    
    def round_us_to_ms(value_us):
    	if value_us < 0:
    		return -((-value_us + 500) // 1000);
    	return (value_us + 500) // 1000;
    
    
    def parse_dta(line):
    	if not line.startswith(DRT_PREFIX):
    		return None
    
    	fields = [
    		value.strip()
    		for value in line[len(DRT_PREFIX):].split(",")
    	];
    
    	if len(fields) < 8:
    		raise ValueError("Incomplete DRT data line: {!r}".format(line));
    
    	onset_ms = int(fields[0]);
    	offset_ms = int(fields[1]);
    	unix = int(fields[2]);
    	trial = int(fields[3]);
    	stimulus = fields[4];
    
    	# Firmware order: RT, then Clicks.
    	rt_ms = int(fields[5]);
    	clicks = int(fields[6]);
    	isi_ms = int(fields[7]);
    
    	# Backward-compatible fallback for older firmware.
    	onset_us = int(fields[8]) if len(fields) > 8 else onset_ms * 1000;
    	offset_us = int(fields[9]) if len(fields) > 9 else offset_ms * 1000;
    
    	if len(fields) > 10:
    		rt_us = int(fields[10]);
    	else:
    		rt_us = -1 if rt_ms < 0 else rt_ms * 1000;
    
    	return {
    		"onset_ms": onset_ms,
    		"offset_ms": offset_ms,
    		"unix": unix,
    		"trial": trial,
    		"stimulus": stimulus,
    		"rt_ms": rt_ms,
    		"clicks": clicks,
    		"isi_ms": isi_ms,
    		"onset_us": onset_us,
    		"offset_us": offset_us,
    		"rt_us": rt_us,
    	}
    
    def reset_epoch_with_rtt(ser, repeats=10, timeout_s=0.5):
    	records = []
    	old_timeout = ser.timeout
    	ser.timeout = 0.01
    
    	try:
    		for i in range(repeats):
    			ser.reset_input_buffer()
    
    			t_send_ns = time.perf_counter_ns()
    			ser.write(b"resettimer\n")
    			ser.flush()
    
    			deadline_ns = t_send_ns + int(timeout_s * 1e9)
    			record = None
    
    			while time.perf_counter_ns() < deadline_ns:
    				raw = ser.readline()
    
    				t_recv_ns = time.perf_counter_ns()
    				t_recv_lsl = local_clock()
    
    				if not raw:
    					continue
    
    				line = raw.decode("utf-8", "replace").rstrip("\r\n")
    
    				if not line.startswith("timer_reset>"):
    					continue
    
    				board_tick = int(line.split(">", 1)[1].strip())
    
    				anchor_pc_ns = (t_send_ns + t_recv_ns) // 2
    				anchor_lsl = (
    					t_recv_lsl
    					- (t_recv_ns - anchor_pc_ns) / 1e9
    				)
    
    				record = {
    					"index": i,
    					"send_ns": t_send_ns,
    					"receive_ns": t_recv_ns,
    					"rtt_ns": t_recv_ns - t_send_ns,
    					"board_tick": board_tick,
    					"anchor_pc_ns": anchor_pc_ns,
    					"anchor_lsl": anchor_lsl,
    				}
    
    				records.append(record)
    				break
    
    				if record is None:
    					raise TimeoutError("No timer_reset response on repetition {}".format(i))
    					
    		rtts_ms = [r["rtt_ns"] / 1e6 for r in records]
    
    		print(
    			"DRT reset RTT: median={:.3f} ms, "
    			"min={:.3f} ms, max={:.3f} ms".format(
    				statistics.median(rtts_ms),
    				min(rtts_ms),
    				max(rtts_ms),
    			)
    		)
    
    		# Only the final reset anchor is valid because each reset
    		# restarts the board's experiment timer.
    		final = records[-1]
    
    		return (
    			final["anchor_pc_ns"],
    			final["anchor_lsl"],
    			records,
    		)
    
    	finally:
    		ser.timeout = old_timeout
    Welcome1 = visual.TextStim(win=win, name='Welcome1',
        text='Welcome! Thank you for participating in this experiment.\n\nThis experiment will consist of two types of task.\n\nThe first task tests your memory. A series of letters will be presented on the screen, one at a time. Your job is to remember the letters. After each letter, your job is to determine whether it is the same as the letter that appeared a certain number of presentations back. We\'ll call this number N. So, if you see a letter "A" and the letter "A" also appeared N presentations back, you should respond "same". If the letter N digits back was not "A", you would not respond.\n\nIn this experiment, N will be either 2 or 4. You will be told the N at the start of a block, and it will not change for that block.\n\nLet\'s do some practice!\n\nTo move to the next screen, press spacebar.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=35.0, wrapWidth=1800.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_23 = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # --- Initialize components for Routine "DRT_Setup" ---
    
    # --- Initialize components for Routine "practice_DRT" ---
    fixation_DRT = visual.TextStim(win=win, name='fixation_DRT',
        text='+',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=25.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "FullFinish" ---
    BlockFinishText_2 = visual.TextStim(win=win, name='BlockFinishText_2',
        text="Congratulations, you've finished the experiment!\n\nPress spacebar to close the experiment.",
        font='Open Sans',
        pos=(0, 0), draggable=False, height=55.0, wrapWidth=1000.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_43 = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Intro" ---
    # create an object to store info about Routine Intro
    Intro = data.Routine(
        name='Intro',
        components=[Welcome1, key_resp_23],
    )
    Intro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_6
    win.mouseVisible=False # hide mouse
    
    # Check which COM port is available and establish initial DRT connection
    # NB Experiment will crash if no DRT found
    detected=False
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    ser = serial.Serial()
    ser.baudrate = 115200  # Try increasing to 115200 for faster communication
    ser.timeout=0.5
    ser.write_timeout=0.5
    ser.exclusive=1
    for p in ports:
        if 'USB Serial Device' in p.description:
            DRT_index=p
            print('DRT Connected to ' + str(p))
            # Connection to port
            ser.port = p.device
            detected=True
            break
    if detected==False:
        #core.quit()
        print("DRT NOT DETECTED")
    if (detected==True):
        print("Detected")
        ser.open()
        time.sleep(.5)
        # Send config commands
        commands = [
            b'HS:VIB,0\n',
            b'HS:LED,0\n',
            b'HS:AUD,0\n',
            b'AUD:L,20\n',
            b'VIB:L,40\n',
            b'LED:L,10\n',
        ]
    
        for cmd in commands:
            ser.reset_input_buffer()  # Clear old data before sending
            start_time = time.perf_counter()  # Record the time before sending
            ser.write(cmd)
            ser.flush()  # Ensure command is fully sent
    
            while True:
                response = ser.readline().decode("utf-8").strip()
                if response == "ACK":  # Wait specifically for "ACK"
                    break
                if time.perf_counter() - start_time > 0.5:  # Failsafe timeout
                    break
                time.sleep(0.05)  # Small delay to prevent CPU overuse
        # Turn light on and off to show the communication is working
        ser.write(b'LED.H\n')
        time.sleep(5)
        ser.write(b'LED.H\n')
        time.sleep(0.5)
        # From run of repeats=100: DRT reset RTT: median=2.595 ms, min=2.067 ms, max=3.102 ms
        initial_timestamp, initial_lsl_timestamp, reset_records = (reset_epoch_with_rtt(ser, repeats=100))
        ser.close()
    # create starting attributes for key_resp_23
    key_resp_23.keys = []
    key_resp_23.rt = []
    _key_resp_23_allKeys = []
    # store start times for Intro
    Intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Intro.tStart = globalClock.getTime(format='float')
    Intro.status = STARTED
    thisExp.addData('Intro.started', Intro.tStart)
    Intro.maxDuration = None
    # keep track of which components have finished
    IntroComponents = Intro.components
    for thisComponent in Intro.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Intro" ---
    thisExp.currentRoutine = Intro
    Intro.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Welcome1* updates
        
        # if Welcome1 is starting this frame...
        if Welcome1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Welcome1.frameNStart = frameN  # exact frame index
            Welcome1.tStart = t  # local t and not account for scr refresh
            Welcome1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Welcome1, 'tStartRefresh')  # time at next scr refresh
            # update status
            Welcome1.status = STARTED
            Welcome1.setAutoDraw(True)
        
        # if Welcome1 is active this frame...
        if Welcome1.status == STARTED:
            # update params
            pass
        
        # *key_resp_23* updates
        waitOnFlip = False
        
        # if key_resp_23 is starting this frame...
        if key_resp_23.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
            # keep track of start time/frame for later
            key_resp_23.frameNStart = frameN  # exact frame index
            key_resp_23.tStart = t  # local t and not account for scr refresh
            key_resp_23.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_23, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_23.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_23.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_23.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_23.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_23.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_23_allKeys.extend(theseKeys)
            if len(_key_resp_23_allKeys):
                key_resp_23.keys = _key_resp_23_allKeys[-1].name  # just the last key pressed
                key_resp_23.rt = _key_resp_23_allKeys[-1].rt
                key_resp_23.duration = _key_resp_23_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Intro,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Intro.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Intro.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Intro.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Intro" ---
    for thisComponent in Intro.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Intro
    Intro.tStop = globalClock.getTime(format='float')
    Intro.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Intro.stopped', Intro.tStop)
    # Run 'End Routine' code from code_6
    outlet.push_sample(
    	["DRTStart"],
    	timestamp=initial_lsl_timestamp,
    )
    print("StartingDRT")
    thisExp.nextEntry()
    # the Routine "Intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "DRT_Setup" ---
    # create an object to store info about Routine DRT_Setup
    DRT_Setup = data.Routine(
        name='DRT_Setup',
        components=[],
    )
    DRT_Setup.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code
    if (detected==True):
    	print("running setup")
    	run_DRT=True
    	DRT_Instance += 1
    	DRT_Trial=1
    
    	if DRT_Instance == 1:
    		ser.open()
    		dat = open('data\%s_%s_%s-DRT.csv' % (expInfo['participant'],expName,data.getDateStr(format="%d-%m-%Y-%H%M")),'w')
    		dat.write("OnsetTimePC_est_ms,OnsetTimeDRT_ms,EndTimeDRT_ms,UnixEpoch,TrialNo,Stimulus,RT_ms,Clicks,ISI_ms,Block,ParticipantID,SerialAge_ms,OnsetTimeDRT_us,EndTimeDRT_us,RT_us\n")
    
    	# Write commands to DRT unit (add sleep between each)
    	if 'Tactile' in Stimulus:
    		commands = [
    		b'WS:LED,0\n',
    		b'WS:VIB,100\n',
    		b'WS:AUD,0\n',
    		b'HMS:2,0\n',
    		b'HMS:1,100\n',
    		b'HMS:3,0\n',
    		b'HMS:0,0\n'
    		]
    	elif 'LED' in Stimulus:
    		commands = [
    		b'WS:LED,100\n',
    		b'WS:VIB,0\n',
    		b'WS:AUD,0\n',
    		b'HMS:2,0\n',
    		b'HMS:1,100\n',
    		b'HMS:3,0\n',
    		b'HMS:0,0\n',
    		b'ISI:L,1000\n',
    		b'ISI:H,1000\n',
    		b'STM:DUR,100\n'
    		]
    	elif 'Auditory' in Stimulus:
    		commands = [
    		b'WS:LED,0\n',
    		b'WS:VIB,0\n',
    		b'WS:AUD,100\n',
    		b'HMS:2,0\n',
    		b'HMS:1,100\n',
    		b'HMS:3,0\n',
    		b'HMS:0,0\n'
    		]
    
    	for cmd in commands:
    		ser.reset_input_buffer()  # Clear old data before sending
    		start_time = time.perf_counter()  # Record the time before sending
    		ser.write(cmd)
    		ser.flush()  # Ensure command is fully sent
    
    		while True:
    			response = ser.readline().decode("utf-8").strip()
    			if response == "ACK":  # Wait specifically for "ACK"
    				break
    			if time.perf_counter() - start_time > 0.5:  # Failsafe timeout
    				break
    			time.sleep(0.05)  # Small delay to prevent CPU overuse
    
    continueRoutine=False
    # store start times for DRT_Setup
    DRT_Setup.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    DRT_Setup.tStart = globalClock.getTime(format='float')
    DRT_Setup.status = STARTED
    thisExp.addData('DRT_Setup.started', DRT_Setup.tStart)
    DRT_Setup.maxDuration = None
    # keep track of which components have finished
    DRT_SetupComponents = DRT_Setup.components
    for thisComponent in DRT_Setup.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "DRT_Setup" ---
    thisExp.currentRoutine = DRT_Setup
    DRT_Setup.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=DRT_Setup,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            DRT_Setup.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if DRT_Setup.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in DRT_Setup.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "DRT_Setup" ---
    for thisComponent in DRT_Setup.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for DRT_Setup
    DRT_Setup.tStop = globalClock.getTime(format='float')
    DRT_Setup.tStopRefresh = tThisFlipGlobal
    thisExp.addData('DRT_Setup.stopped', DRT_Setup.tStop)
    thisExp.nextEntry()
    # the Routine "DRT_Setup" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "practice_DRT" ---
    # create an object to store info about Routine practice_DRT
    practice_DRT = data.Routine(
        name='practice_DRT',
        components=[fixation_DRT],
    )
    practice_DRT.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from DRT_Code_Practice
    Block=0
    if (detected==True):
    	print("Beginning Routine")
    	ser.write(b'resettimer\n')
    	(initial_timestamp,
    	  initial_lsl_timestamp,
    	  reset_board_tick,
    	  reset_rtt_ns,
    	) = reset_drt_epoch(ser)
    
    	print("DRT reset RTT: {:.3f} ms".format(reset_rtt_ns / 1e6))
    
    	outlet.push_sample(
    	  ["DRTStart"],
    	  timestamp=initial_lsl_timestamp,
    	)
    # store start times for practice_DRT
    practice_DRT.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    practice_DRT.tStart = globalClock.getTime(format='float')
    practice_DRT.status = STARTED
    thisExp.addData('practice_DRT.started', practice_DRT.tStart)
    practice_DRT.maxDuration = None
    # keep track of which components have finished
    practice_DRTComponents = practice_DRT.components
    for thisComponent in practice_DRT.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "practice_DRT" ---
    thisExp.currentRoutine = practice_DRT
    practice_DRT.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 300.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation_DRT* updates
        
        # if fixation_DRT is starting this frame...
        if fixation_DRT.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixation_DRT.frameNStart = frameN  # exact frame index
            fixation_DRT.tStart = t  # local t and not account for scr refresh
            fixation_DRT.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_DRT, 'tStartRefresh')  # time at next scr refresh
            # update status
            fixation_DRT.status = STARTED
            fixation_DRT.setAutoDraw(True)
        
        # if fixation_DRT is active this frame...
        if fixation_DRT.status == STARTED:
            # update params
            pass
        
        # if fixation_DRT is stopping this frame...
        if fixation_DRT.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fixation_DRT.tStartRefresh + 300-frameTolerance:
                # keep track of stop time/frame for later
                fixation_DRT.tStop = t  # not accounting for scr refresh
                fixation_DRT.tStopRefresh = tThisFlipGlobal  # on global time
                fixation_DRT.frameNStop = frameN  # exact frame index
                # update status
                fixation_DRT.status = FINISHED
                fixation_DRT.setAutoDraw(False)
        # Run 'Each Frame' code from DRT_Code_Practice
        if (detected==True):
        	# This loop collects each DRT trial and saves to the csv we created
        	CurrentTimer=(time.perf_counter_ns()-initial_timestamp) // 1000000000
        	if frameN == FrameRate:
        		ser.write(b"start\n")
        		ser.flush()
        		drt_running = True
        
        	if drt_running and ser.in_waiting and CurrentTimer<=duration:
        		raw = ser.readline()
        
        		read_ns = time.perf_counter_ns()
        		read_lsl = local_clock()
        
        		line = raw.decode("utf-8", "replace").rstrip("\r\n")
        
        		if line.startswith(DRT_PREFIX):
        			record = parse_dta(line)
        
        			pc_elapsed_us = (
        				read_ns - initial_timestamp
        			) // 1000
        
        			# This is serial/read age, not a calibrated clock offset.
        			serial_age_us = pc_elapsed_us - record["onset_us"]
        
        			# Equivalent to the old correction, but now without ontime/lag.
        			onset_timestamp = (
        				read_lsl - serial_age_us / 1_000_000
        			)
        
        			outlet.push_sample(
        				["{}_ON".format(record["stimulus"])],
        				timestamp=onset_timestamp,
        			)
        
        			if record["rt_us"] >= 0:
        				outlet.push_sample(
        					["{}_RESP".format(record["stimulus"])],
        					timestamp=(
        						onset_timestamp
        						+ record["rt_us"] / 1_000_000
        					),
        				)
        			else:
        				# Preserves current behavior: MISS is placed at offset_us.
        				miss_timestamp = (
        					read_lsl
        					- (
        						pc_elapsed_us - record["offset_us"]
        					) / 1_000_000
        				)
        
        				outlet.push_sample(
        					["{}_MISS".format(record["stimulus"])],
        					timestamp=miss_timestamp,
        				)
        
        			ontime = round_us_to_ms(record["onset_us"])
        			lag = round_us_to_ms(serial_age_us)
        
        			dat.write(
        				"{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
        					ontime,
        					record["onset_ms"],
        					record["offset_ms"],
        					record["unix"],
        					record["trial"],
        					record["stimulus"],
        					record["rt_ms"],
        					record["clicks"],
        					record["isi_ms"],
        					Block,
        					expInfo["participant"],
        					lag,
        					record["onset_us"],
        					record["offset_us"],
        					record["rt_us"],
        				)
        			)
        
        		elif line.startswith("timer_check>"):
        			print(line)
        if CurrentTimer>duration:
        	continueRoutine=False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=practice_DRT,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            practice_DRT.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if practice_DRT.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in practice_DRT.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "practice_DRT" ---
    for thisComponent in practice_DRT.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for practice_DRT
    practice_DRT.tStop = globalClock.getTime(format='float')
    practice_DRT.tStopRefresh = tThisFlipGlobal
    thisExp.addData('practice_DRT.stopped', practice_DRT.tStop)
    # Run 'End Routine' code from DRT_Code_Practice
    if (detected==True):
        ser.write(b'stop\n')
    drt_running=False
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if practice_DRT.maxDurationReached:
        routineTimer.addTime(-practice_DRT.maxDuration)
    elif practice_DRT.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-300.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "FullFinish" ---
    # create an object to store info about Routine FullFinish
    FullFinish = data.Routine(
        name='FullFinish',
        components=[BlockFinishText_2, key_resp_43],
    )
    FullFinish.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_43
    key_resp_43.keys = []
    key_resp_43.rt = []
    _key_resp_43_allKeys = []
    # Run 'Begin Routine' code from code_15
    ser.close()
    outlet2.push_sample([f"ExpEnd"],timestamp=local_clock())
    # store start times for FullFinish
    FullFinish.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    FullFinish.tStart = globalClock.getTime(format='float')
    FullFinish.status = STARTED
    thisExp.addData('FullFinish.started', FullFinish.tStart)
    FullFinish.maxDuration = None
    # keep track of which components have finished
    FullFinishComponents = FullFinish.components
    for thisComponent in FullFinish.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "FullFinish" ---
    thisExp.currentRoutine = FullFinish
    FullFinish.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *BlockFinishText_2* updates
        
        # if BlockFinishText_2 is starting this frame...
        if BlockFinishText_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            BlockFinishText_2.frameNStart = frameN  # exact frame index
            BlockFinishText_2.tStart = t  # local t and not account for scr refresh
            BlockFinishText_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(BlockFinishText_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            BlockFinishText_2.status = STARTED
            BlockFinishText_2.setAutoDraw(True)
        
        # if BlockFinishText_2 is active this frame...
        if BlockFinishText_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_43* updates
        waitOnFlip = False
        
        # if key_resp_43 is starting this frame...
        if key_resp_43.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_43.frameNStart = frameN  # exact frame index
            key_resp_43.tStart = t  # local t and not account for scr refresh
            key_resp_43.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_43, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_43.started')
            # update status
            key_resp_43.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_43.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_43.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_43.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_43.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_43_allKeys.extend(theseKeys)
            if len(_key_resp_43_allKeys):
                key_resp_43.keys = _key_resp_43_allKeys[-1].name  # just the last key pressed
                key_resp_43.rt = _key_resp_43_allKeys[-1].rt
                key_resp_43.duration = _key_resp_43_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=FullFinish,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            FullFinish.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if FullFinish.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in FullFinish.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "FullFinish" ---
    for thisComponent in FullFinish.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for FullFinish
    FullFinish.tStop = globalClock.getTime(format='float')
    FullFinish.tStopRefresh = tThisFlipGlobal
    thisExp.addData('FullFinish.stopped', FullFinish.tStop)
    thisExp.nextEntry()
    # the Routine "FullFinish" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='comma')


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
