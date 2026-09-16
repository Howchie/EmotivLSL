#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.3),
    on September 15, 2026, at 10:28
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
psychopyVersion = '2026.1.3'
expName = 'oddball'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
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
_winSize = [1024,768]
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
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\howch\\OneDrive\\Documents\\Oddball\\oddball.py',
        savePickle=True, saveWideText=True,
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
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
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
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
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
    # Run 'Begin Experiment' code from intro_code
    win.mouseVisible=False
    from pylsl import StreamInfo, StreamOutlet, local_clock
    import psychtoolbox as ptb
    from psychopy import sound
    import random as random
    import numpy as np
    from psychopy import prefs
    prefs.hardware['audioLib'] = ['ptb']
    prefs.hardware['audioLatencyMode'] = '3'
    # Create marker stream for Lab Streaming Layer
    info = StreamInfo(name='PsychoPy Markers', type='Markers', channel_count=1,nominal_srate=0, 
                      channel_format='string', source_id=expInfo['participant'])
    outlet = StreamOutlet(info)  # Broadcast the stream.
    mySound = sound.Sound(780,volume=1,stereo=False,secs=0.1)
    text = visual.TextStim(win=win, name='text',
        text='Hello welcome to the experiment',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "CloseEyes" ---
    text5 = visual.TextStim(win=win, name='text5',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=0.0);
    key_resp2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "OpenEyes" ---
    text4 = visual.TextStim(win=win, name='text4',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=0.0);
    key_resp2_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "OddballIntro" ---
    # Run 'Begin Experiment' code from code
    win.mouseVisible=False
    from pylsl import StreamInfo, StreamOutlet, local_clock
    import psychtoolbox as ptb
    from psychopy import sound
    import random as random
    import numpy as np
    # Create marker stream for Lab Streaming Layer
    info = StreamInfo(name='PsychoPy Markers', type='Markers', channel_count=1,nominal_srate=0, 
                      channel_format='string', source_id=expInfo['participant'])
    outlet = StreamOutlet(info)  # Broadcast the stream.
    mySound = sound.Sound(780,volume=1,stereo=False,secs=0.1)
    text6 = visual.TextStim(win=win, name='text6',
        text='The following block requires you to watch this patch of random noise.\n\nTones will play but you should ignore them.',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "oddball_trial" ---
    dots = visual.DotStim(
        win=win, name='dots',
        nDots=100, dotSize=5.0,
        speed=0.01, dir=0.0, coherence=0.0,
        fieldPos=[0,0], fieldSize=1.0, fieldAnchor='center', fieldShape='circle',
        signalDots='different', noiseDots='walk',dotLife=3.0,
        color=[1.0,1.0,1.0], colorSpace='rgb', opacity=None,
        depth=-1.0)
    
    # --- Initialize components for Routine "ChoiceIntro" ---
    # Run 'Begin Experiment' code from code_2
    win.mouseVisible=False
    from pylsl import StreamInfo, StreamOutlet, local_clock
    import psychtoolbox as ptb
    from psychopy import sound
    import random as random
    import numpy as np
    # Create marker stream for Lab Streaming Layer
    info = StreamInfo(name='PsychoPy Markers', type='Markers', channel_count=1,nominal_srate=0, 
                      channel_format='string', source_id=expInfo['participant'])
    outlet = StreamOutlet(info)  # Broadcast the stream.
    mySound = sound.Sound(780,volume=1,stereo=False,secs=0.1)
    text6_2 = visual.TextStim(win=win, name='text6_2',
        text="The following block requires you to discriminate between the two tones.\n\nWhen you heart the high pitch tone from the previous block, press 'space'.\nWhen you hear the low pitch tone from the previous block, do not press any button.\n\nPress space to begin.",
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_3 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "choice_trial" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text='+',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    choice_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "End" ---
    outro = visual.TextStim(win=win, name='outro',
        text='All finished. Thanks!',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
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
        components=[text, key_resp],
    )
    Intro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for Intro
    Intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Intro.tStart = globalClock.getTime(format='float')
    Intro.status = STARTED
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
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            key_resp.clock.reset()  # now t=0
        if key_resp.status == STARTED:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
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
    thisExp.nextEntry()
    # the Routine "Intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    eyes = data.TrialHandler2(
        name='eyes',
        nReps=4, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
        isTrials=False, 
    )
    thisExp.addLoop(eyes)  # add the loop to the experiment
    thisEye = eyes.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisEye.rgb)
    if thisEye != None:
        for paramName in thisEye:
            globals()[paramName] = thisEye[paramName]
    
    for thisEye in eyes:
        eyes.status = STARTED
        if hasattr(thisEye, 'status'):
            thisEye.status = STARTED
        currentLoop = eyes
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisEye.rgb)
        if thisEye != None:
            for paramName in thisEye:
                globals()[paramName] = thisEye[paramName]
        
        # --- Prepare to start Routine "CloseEyes" ---
        # create an object to store info about Routine CloseEyes
        CloseEyes = data.Routine(
            name='CloseEyes',
            components=[text5, key_resp2],
        )
        CloseEyes.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text5.setText('Close your eyes for 60 seconds.')
        # Run 'Begin Routine' code from code4_2
        outlet.push_sample(["CloseEyes_Start"],timestamp=local_clock())
        # create starting attributes for key_resp2
        key_resp2.keys = []
        key_resp2.rt = []
        _key_resp2_allKeys = []
        # store start times for CloseEyes
        CloseEyes.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        CloseEyes.tStart = globalClock.getTime(format='float')
        CloseEyes.status = STARTED
        CloseEyes.maxDuration = None
        # keep track of which components have finished
        CloseEyesComponents = CloseEyes.components
        for thisComponent in CloseEyes.components:
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
        
        # --- Run Routine "CloseEyes" ---
        thisExp.currentRoutine = CloseEyes
        CloseEyes.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisEye, 'status') and thisEye.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text5* updates
            
            # if text5 is starting this frame...
            if text5.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                text5.frameNStart = frameN  # exact frame index
                text5.tStart = t  # local t and not account for scr refresh
                text5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text5, 'tStartRefresh')  # time at next scr refresh
                # update status
                text5.status = STARTED
                text5.setAutoDraw(True)
            
            # if text5 is active this frame...
            if text5.status == STARTED:
                # update params
                pass
            
            # if text5 is stopping this frame...
            if text5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text5.tStartRefresh + 70-frameTolerance:
                    # keep track of stop time/frame for later
                    text5.tStop = t  # not accounting for scr refresh
                    text5.tStopRefresh = tThisFlipGlobal  # on global time
                    text5.frameNStop = frameN  # exact frame index
                    # update status
                    text5.status = FINISHED
                    text5.setAutoDraw(False)
            
            # *key_resp2* updates
            
            # if key_resp2 is starting this frame...
            if key_resp2.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                key_resp2.frameNStart = frameN  # exact frame index
                key_resp2.tStart = t  # local t and not account for scr refresh
                key_resp2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp2, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp2.status = STARTED
                # keyboard checking is just starting
                key_resp2.clock.reset()  # now t=0
            if key_resp2.status == STARTED:
                theseKeys = key_resp2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp2_allKeys.extend(theseKeys)
                if len(_key_resp2_allKeys):
                    key_resp2.keys = _key_resp2_allKeys[-1].name  # just the last key pressed
                    key_resp2.rt = _key_resp2_allKeys[-1].rt
                    key_resp2.duration = _key_resp2_allKeys[-1].duration
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
                    currentRoutine=CloseEyes,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                CloseEyes.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if CloseEyes.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in CloseEyes.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "CloseEyes" ---
        for thisComponent in CloseEyes.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for CloseEyes
        CloseEyes.tStop = globalClock.getTime(format='float')
        CloseEyes.tStopRefresh = tThisFlipGlobal
        # Run 'End Routine' code from code4_2
        outlet.push_sample(["CloseEyes_End"],timestamp=local_clock())
        # the Routine "CloseEyes" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "OpenEyes" ---
        # create an object to store info about Routine OpenEyes
        OpenEyes = data.Routine(
            name='OpenEyes',
            components=[text4, key_resp2_2],
        )
        OpenEyes.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text4.setText('Open your eyes for 60 seconds looking at this fixation cross:\n\nx')
        # Run 'Begin Routine' code from code3
        outlet.push_sample(["OpenEyes_Start"],timestamp=local_clock())
        # create starting attributes for key_resp2_2
        key_resp2_2.keys = []
        key_resp2_2.rt = []
        _key_resp2_2_allKeys = []
        # store start times for OpenEyes
        OpenEyes.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        OpenEyes.tStart = globalClock.getTime(format='float')
        OpenEyes.status = STARTED
        OpenEyes.maxDuration = None
        # keep track of which components have finished
        OpenEyesComponents = OpenEyes.components
        for thisComponent in OpenEyes.components:
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
        
        # --- Run Routine "OpenEyes" ---
        thisExp.currentRoutine = OpenEyes
        OpenEyes.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisEye, 'status') and thisEye.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text4* updates
            
            # if text4 is starting this frame...
            if text4.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                text4.frameNStart = frameN  # exact frame index
                text4.tStart = t  # local t and not account for scr refresh
                text4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text4, 'tStartRefresh')  # time at next scr refresh
                # update status
                text4.status = STARTED
                text4.setAutoDraw(True)
            
            # if text4 is active this frame...
            if text4.status == STARTED:
                # update params
                pass
            
            # if text4 is stopping this frame...
            if text4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text4.tStartRefresh + 70-frameTolerance:
                    # keep track of stop time/frame for later
                    text4.tStop = t  # not accounting for scr refresh
                    text4.tStopRefresh = tThisFlipGlobal  # on global time
                    text4.frameNStop = frameN  # exact frame index
                    # update status
                    text4.status = FINISHED
                    text4.setAutoDraw(False)
            
            # *key_resp2_2* updates
            
            # if key_resp2_2 is starting this frame...
            if key_resp2_2.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                key_resp2_2.frameNStart = frameN  # exact frame index
                key_resp2_2.tStart = t  # local t and not account for scr refresh
                key_resp2_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp2_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp2_2.status = STARTED
                # keyboard checking is just starting
                key_resp2_2.clock.reset()  # now t=0
            if key_resp2_2.status == STARTED:
                theseKeys = key_resp2_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp2_2_allKeys.extend(theseKeys)
                if len(_key_resp2_2_allKeys):
                    key_resp2_2.keys = _key_resp2_2_allKeys[-1].name  # just the last key pressed
                    key_resp2_2.rt = _key_resp2_2_allKeys[-1].rt
                    key_resp2_2.duration = _key_resp2_2_allKeys[-1].duration
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
                    currentRoutine=OpenEyes,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                OpenEyes.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if OpenEyes.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in OpenEyes.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "OpenEyes" ---
        for thisComponent in OpenEyes.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for OpenEyes
        OpenEyes.tStop = globalClock.getTime(format='float')
        OpenEyes.tStopRefresh = tThisFlipGlobal
        # Run 'End Routine' code from code3
        outlet.push_sample(["OpenEyes_End"],timestamp=local_clock())
        # the Routine "OpenEyes" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisEye as finished
        if hasattr(thisEye, 'status'):
            thisEye.status = FINISHED
        # if awaiting a pause, pause now
        if eyes.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            eyes.status = STARTED
    # completed 4 repeats of 'eyes'
    eyes.status = FINISHED
    
    
    # --- Prepare to start Routine "OddballIntro" ---
    # create an object to store info about Routine OddballIntro
    OddballIntro = data.Routine(
        name='OddballIntro',
        components=[text6, key_resp_2],
    )
    OddballIntro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for OddballIntro
    OddballIntro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    OddballIntro.tStart = globalClock.getTime(format='float')
    OddballIntro.status = STARTED
    OddballIntro.maxDuration = None
    # keep track of which components have finished
    OddballIntroComponents = OddballIntro.components
    for thisComponent in OddballIntro.components:
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
    
    # --- Run Routine "OddballIntro" ---
    thisExp.currentRoutine = OddballIntro
    OddballIntro.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text6* updates
        
        # if text6 is starting this frame...
        if text6.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            text6.frameNStart = frameN  # exact frame index
            text6.tStart = t  # local t and not account for scr refresh
            text6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text6, 'tStartRefresh')  # time at next scr refresh
            # update status
            text6.status = STARTED
            text6.setAutoDraw(True)
        
        # if text6 is active this frame...
        if text6.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            key_resp_2.clock.reset()  # now t=0
        if key_resp_2.status == STARTED:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
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
                currentRoutine=OddballIntro,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            OddballIntro.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if OddballIntro.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in OddballIntro.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "OddballIntro" ---
    for thisComponent in OddballIntro.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for OddballIntro
    OddballIntro.tStop = globalClock.getTime(format='float')
    OddballIntro.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "OddballIntro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    oddball = data.TrialHandler2(
        name='oddball',
        nReps=20, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('oddball.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(oddball)  # add the loop to the experiment
    thisOddball = oddball.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisOddball.rgb)
    if thisOddball != None:
        for paramName in thisOddball:
            globals()[paramName] = thisOddball[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisOddball in oddball:
        oddball.status = STARTED
        if hasattr(thisOddball, 'status'):
            thisOddball.status = STARTED
        currentLoop = oddball
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisOddball.rgb)
        if thisOddball != None:
            for paramName in thisOddball:
                globals()[paramName] = thisOddball[paramName]
        
        # --- Prepare to start Routine "oddball_trial" ---
        # create an object to store info about Routine oddball_trial
        oddball_trial = data.Routine(
            name='oddball_trial',
            components=[dots],
        )
        oddball_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_oddball
        isi = np.random.uniform(low=1,high=1.5)
        soa = .2
        mySound.setSound(freq)
        now = ptb.GetSecs()
        mySound.play(when=now+soa)
        outlet.push_sample([f"Oddball-{stim}"],timestamp=local_clock()+soa)
        thisExp.addData('isi',isi)
        thisExp.addData('soa',soa)
        dots.refreshDots()
        # store start times for oddball_trial
        oddball_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        oddball_trial.tStart = globalClock.getTime(format='float')
        oddball_trial.status = STARTED
        oddball_trial.maxDuration = None
        # keep track of which components have finished
        oddball_trialComponents = oddball_trial.components
        for thisComponent in oddball_trial.components:
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
        
        # --- Run Routine "oddball_trial" ---
        thisExp.currentRoutine = oddball_trial
        oddball_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisOddball, 'status') and thisOddball.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dots* updates
            
            # if dots is starting this frame...
            if dots.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                dots.frameNStart = frameN  # exact frame index
                dots.tStart = t  # local t and not account for scr refresh
                dots.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dots, 'tStartRefresh')  # time at next scr refresh
                # update status
                dots.status = STARTED
                dots.setAutoDraw(True)
            
            # if dots is active this frame...
            if dots.status == STARTED:
                # update params
                pass
            
            # if dots is stopping this frame...
            if dots.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dots.tStartRefresh + isi-frameTolerance:
                    # keep track of stop time/frame for later
                    dots.tStop = t  # not accounting for scr refresh
                    dots.tStopRefresh = tThisFlipGlobal  # on global time
                    dots.frameNStop = frameN  # exact frame index
                    # update status
                    dots.status = FINISHED
                    dots.setAutoDraw(False)
            
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
                    currentRoutine=oddball_trial,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                oddball_trial.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if oddball_trial.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in oddball_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "oddball_trial" ---
        for thisComponent in oddball_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for oddball_trial
        oddball_trial.tStop = globalClock.getTime(format='float')
        oddball_trial.tStopRefresh = tThisFlipGlobal
        # the Routine "oddball_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisOddball as finished
        if hasattr(thisOddball, 'status'):
            thisOddball.status = FINISHED
        # if awaiting a pause, pause now
        if oddball.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            oddball.status = STARTED
        thisExp.nextEntry()
        
    # completed 20 repeats of 'oddball'
    oddball.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "ChoiceIntro" ---
    # create an object to store info about Routine ChoiceIntro
    ChoiceIntro = data.Routine(
        name='ChoiceIntro',
        components=[text6_2, key_resp_3],
    )
    ChoiceIntro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_3
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # store start times for ChoiceIntro
    ChoiceIntro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ChoiceIntro.tStart = globalClock.getTime(format='float')
    ChoiceIntro.status = STARTED
    ChoiceIntro.maxDuration = None
    # keep track of which components have finished
    ChoiceIntroComponents = ChoiceIntro.components
    for thisComponent in ChoiceIntro.components:
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
    
    # --- Run Routine "ChoiceIntro" ---
    thisExp.currentRoutine = ChoiceIntro
    ChoiceIntro.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text6_2* updates
        
        # if text6_2 is starting this frame...
        if text6_2.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            text6_2.frameNStart = frameN  # exact frame index
            text6_2.tStart = t  # local t and not account for scr refresh
            text6_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text6_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text6_2.status = STARTED
            text6_2.setAutoDraw(True)
        
        # if text6_2 is active this frame...
        if text6_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            key_resp_3.clock.reset()  # now t=0
        if key_resp_3.status == STARTED:
            theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
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
                currentRoutine=ChoiceIntro,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            ChoiceIntro.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if ChoiceIntro.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in ChoiceIntro.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ChoiceIntro" ---
    for thisComponent in ChoiceIntro.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ChoiceIntro
    ChoiceIntro.tStop = globalClock.getTime(format='float')
    ChoiceIntro.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "ChoiceIntro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    gng = data.TrialHandler2(
        name='gng',
        nReps=10, 
        method='fullRandom', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('gng.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(gng)  # add the loop to the experiment
    thisGng = gng.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisGng.rgb)
    if thisGng != None:
        for paramName in thisGng:
            globals()[paramName] = thisGng[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisGng in gng:
        gng.status = STARTED
        if hasattr(thisGng, 'status'):
            thisGng.status = STARTED
        currentLoop = gng
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisGng.rgb)
        if thisGng != None:
            for paramName in thisGng:
                globals()[paramName] = thisGng[paramName]
        
        # --- Prepare to start Routine "choice_trial" ---
        # create an object to store info about Routine choice_trial
        choice_trial = data.Routine(
            name='choice_trial',
            components=[fixation, choice_resp],
        )
        choice_trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code2
        isi = np.random.uniform(low=1,high=1.5)
        soa=.1+np.random.exponential(scale=.25)
        mySound.setSound(freq)
        now = ptb.GetSecs()
        mySound.play(when=now+soa)
        outlet.push_sample([f"GnG-{stim}"],timestamp=local_clock()+soa)
        thisExp.addData('isi',isi)
        thisExp.addData('soa',soa)
        # create starting attributes for choice_resp
        choice_resp.keys = []
        choice_resp.rt = []
        _choice_resp_allKeys = []
        # store start times for choice_trial
        choice_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        choice_trial.tStart = globalClock.getTime(format='float')
        choice_trial.status = STARTED
        choice_trial.maxDuration = None
        # keep track of which components have finished
        choice_trialComponents = choice_trial.components
        for thisComponent in choice_trial.components:
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
        
        # --- Run Routine "choice_trial" ---
        thisExp.currentRoutine = choice_trial
        choice_trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisGng, 'status') and thisGng.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + soa-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *choice_resp* updates
            waitOnFlip = False
            
            # if choice_resp is starting this frame...
            if choice_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                choice_resp.frameNStart = frameN  # exact frame index
                choice_resp.tStart = t  # local t and not account for scr refresh
                choice_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(choice_resp, 'tStartRefresh')  # time at next scr refresh
                # update status
                choice_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(choice_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(choice_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if choice_resp is stopping this frame...
            if choice_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > choice_resp.tStartRefresh + isi-frameTolerance:
                    # keep track of stop time/frame for later
                    choice_resp.tStop = t  # not accounting for scr refresh
                    choice_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    choice_resp.frameNStop = frameN  # exact frame index
                    # update status
                    choice_resp.status = FINISHED
                    choice_resp.status = FINISHED
            if choice_resp.status == STARTED and not waitOnFlip:
                theseKeys = choice_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _choice_resp_allKeys.extend(theseKeys)
                if len(_choice_resp_allKeys):
                    choice_resp.keys = _choice_resp_allKeys[-1].name  # just the last key pressed
                    choice_resp.rt = _choice_resp_allKeys[-1].rt
                    choice_resp.duration = _choice_resp_allKeys[-1].duration
                    # was this correct?
                    if (choice_resp.keys == str(correct)) or (choice_resp.keys == correct):
                        choice_resp.corr = 1
                    else:
                        choice_resp.corr = 0
            
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
                    currentRoutine=choice_trial,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                choice_trial.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if choice_trial.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in choice_trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "choice_trial" ---
        for thisComponent in choice_trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for choice_trial
        choice_trial.tStop = globalClock.getTime(format='float')
        choice_trial.tStopRefresh = tThisFlipGlobal
        # check responses
        if choice_resp.keys in ['', [], None]:  # No response was made
            choice_resp.keys = None
            # was no response the correct answer?!
            if str(correct).lower() == 'none':
               choice_resp.corr = 1;  # correct non-response
            else:
               choice_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for gng (TrialHandler)
        gng.addData('choice_resp.keys',choice_resp.keys)
        gng.addData('choice_resp.corr', choice_resp.corr)
        if choice_resp.keys != None:  # we had a response
            gng.addData('choice_resp.rt', choice_resp.rt)
            gng.addData('choice_resp.duration', choice_resp.duration)
        # the Routine "choice_trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisGng as finished
        if hasattr(thisGng, 'status'):
            thisGng.status = FINISHED
        # if awaiting a pause, pause now
        if gng.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            gng.status = STARTED
        thisExp.nextEntry()
        
    # completed 10 repeats of 'gng'
    gng.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "End" ---
    # create an object to store info about Routine End
    End = data.Routine(
        name='End',
        components=[outro],
    )
    End.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code4
    outlet.push_sample(["Oddball_End"],timestamp=local_clock())
    # store start times for End
    End.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    End.tStart = globalClock.getTime(format='float')
    End.status = STARTED
    End.maxDuration = None
    # keep track of which components have finished
    EndComponents = End.components
    for thisComponent in End.components:
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
    
    # --- Run Routine "End" ---
    thisExp.currentRoutine = End
    End.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *outro* updates
        
        # if outro is starting this frame...
        if outro.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            outro.frameNStart = frameN  # exact frame index
            outro.tStart = t  # local t and not account for scr refresh
            outro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(outro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'outro.started')
            # update status
            outro.status = STARTED
            outro.setAutoDraw(True)
        
        # if outro is active this frame...
        if outro.status == STARTED:
            # update params
            pass
        
        # if outro is stopping this frame...
        if outro.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > outro.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                outro.tStop = t  # not accounting for scr refresh
                outro.tStopRefresh = tThisFlipGlobal  # on global time
                outro.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'outro.stopped')
                # update status
                outro.status = FINISHED
                outro.setAutoDraw(False)
        
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
                currentRoutine=End,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            End.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if End.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in End.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End" ---
    for thisComponent in End.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for End
    End.tStop = globalClock.getTime(format='float')
    End.tStopRefresh = tThisFlipGlobal
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if End.maxDurationReached:
        routineTimer.addTime(-End.maxDuration)
    elif End.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
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
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


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
