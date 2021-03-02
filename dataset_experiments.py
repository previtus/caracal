# Fast code to extract event samples from the dataset given the classical pipeline of Caracal.
# We get samples which contain events (due to coherence criteria) and events without any events.

import os
import sys

from shumba import shumbacore
from shumba import audiocore
from shumba import extract
from shumba import match
from shumba import position

import pylab as plt
import numpy as np
from matplotlib.patches import Rectangle

################
"""
# we create our logger objects and embed them into our world.
# the size of our world. note that the origin (0,0) is defined by
# the position of the lower left node
worldExtents = [-1500,3000,-1500,3000]
world = shumbacore.World(worldExtents,sample_rate=44100)
coarse_timestamps = [1528419100,1528419200,1528419300,1528419400,1528419500]
# Where the files live
logger_directory = "/home/vitek/Vitek/python_codes/ox_audio_analysis_animals/DATA_BuffaloKillResampled/"
logger_filepattern =  "{:02d}/hq_{:d}.wav"
# Coordinates of the loggers
logger7pos = shumbacore.Position(-21.7278,29.8843,'LatLong')
logger8pos = shumbacore.Position(-21.7237,29.8819,'LatLong')
logger10pos = shumbacore.Position(-21.7317,29.8876,'LatLong')
logger14pos = shumbacore.Position(-21.7359,29.8777,'LatLong')
# Setup the loggers themselves
logger7 = shumbacore.Logger(id=7,name="7",position=logger7pos,directory=logger_directory,filetemplate=logger_filepattern)
logger8 = shumbacore.Logger(id=8,name="8",position=logger8pos,directory=logger_directory,filetemplate=logger_filepattern)
logger10 = shumbacore.Logger(id=10,name="10",position=logger10pos,directory=logger_directory,filetemplate=logger_filepattern)
logger14 = shumbacore.Logger(id=14,name="14",position=logger14pos,directory=logger_directory,filetemplate=logger_filepattern)
loggerlist = [logger7,logger8,logger10,logger14]
# Add the loggers to the world
for logger in loggerlist:
    world.addLogger(logger)

lat = -21.7278
lon = 29.8797
targetpos = shumbacore.Position(lat,lon,'LatLong')
target = shumbacore.Target(0,"KillLoc",targetpos)
world.addTarget(target)
"""

logger_directory = "/home/vitek/Vitek/python_codes/ox_audio_analysis_animals/DATA_BuffaloKillResampled/"
logger_filepattern =  "{:02d}/hq_{:d}.wav"
logger8pos = shumbacore.Position(-21.7237,29.8819,'LatLong')
logger8 = shumbacore.Logger(id=8,name="8",position=logger8pos,directory=logger_directory,filetemplate=logger_filepattern)

#################

selected_coarse_timestamp = 1528419100
selected_logger = logger8

rerun = True
if rerun:
    # Get all positive matches for events ~ "has event"
    coherence_threshold = 5.9  # 5->still lot, 6-> none
    merge = 44 # 40 is close to having no overlap in between two 4 sec events
    bigger_than_coherence = lambda metric, coherence_thr: metric > coherence_thr

    ev = extract.CoherentEventSegmenter(coherence_threshold=coherence_threshold,merge=merge,event_length=4.0, metric_comparison_function_over=bigger_than_coherence)
    bf, events = ev.extractEvents(selected_logger, selected_coarse_timestamp)
    print("Strong events: ", len(events), "events have coherence > ",coherence_threshold)

    # Get all negative matches for events ~ "doesnt have event"
    coherence_threshold = 3.0
    merge = 44
    smaller_than_coherence = lambda metric, coherence_thr: metric < coherence_thr

    non_ev = extract.CoherentEventSegmenter(coherence_threshold=coherence_threshold,merge=merge,event_length=4.0, metric_comparison_function_over=smaller_than_coherence)
    _, non_events = non_ev.extractEvents(selected_logger, selected_coarse_timestamp)
    print("Non events: ", len(non_events), "'non-events' have coherence < ",coherence_threshold)

## Load from objects:
rawAudio = audiocore.AudioFile(selected_logger).loadAtTime(selected_coarse_timestamp)
rawAudio = rawAudio.copy()

print("rawAudio.shape", rawAudio.shape)

### Possibly evaluate overlaps
"""
ranges_events = []
ranges_nonevents = []
for event_i in range(len(events)):
    t_start_orig = events[event_i].event_time
    t_end_orig = t_start_orig + events[event_i].event_length
    ranges_events.append([t_start_orig,t_end_orig]) # they are also ordered...
for non_event_i in range(len(non_events)): # x1,x2 <?> y1,y2
    t_start_orig = non_events[non_event_i].event_time
    t_end_orig = t_start_orig + non_events[non_event_i].event_length
    ranges_nonevents.append([t_start_orig,t_end_orig]) # they are also ordered...

def is_overlapping(x1,x2,y1,y2):
    return max(x1,y1) <= min(x2,y2)

def arrays_overlapping(main_array, secondary_array):
    # arrays contain [start_event_time, end_event_time]
    # returns boolean flags for items in the secondary_array non overlapped by items in the main_array
    nonoverlapping_nonevent_flags = []

    overlapping = False
    for rx in secondary_array:
        x1, x2 = rx

        for ry in main_array:
            y1, y2 = ry
            if is_overlapping(x1,x2,y1,y2): # one overlaping is enough
                overlapping = True
                break
            if y2 > x2: # .. we could also stop early for efficiency here
                break
        nonoverlapping_nonevent_flags.append(not overlapping)
    return nonoverlapping_nonevent_flags

nonoverlapping_nonevent_flags = arrays_overlapping(ranges_events, ranges_nonevents)
"""

def is_overlapping(x1,x2,y1,y2):
    return max(x1,y1) <= min(x2,y2)

# ranges_events contain [start_event_time, end_event_time]
# nonoverlapping_nonevent_flags has boolean flags for items in the non_events non overlapped by items in the events
ranges_events = []
nonoverlapping_nonevent_flags = []

for event_i in range(len(events)):
    t_start_orig = events[event_i].event_time
    t_end_orig = t_start_orig + events[event_i].event_length
    ranges_events.append([t_start_orig,t_end_orig]) # they are also ordered...

for non_event_i in range(len(non_events)): # x1,x2 <?> y1,y2
    x1 = non_events[non_event_i].event_time
    x2 = x1 + non_events[non_event_i].event_length

    overlapping = False
    for r in ranges_events:
        y1, y2 = r
        if is_overlapping(x1,x2,y1,y2): # one overlaping is enough
            overlapping = True
            break
        if y2 > x2: # .. we could also stop early for efficiency here
            break
    nonoverlapping_nonevent_flags.append(not overlapping)

def draw_bboxes_for_events(events, ax, boolean_mask=None):
    orig_xlim = ax.get_xlim()
    ### Mark down events
    for event_i in range(len(events)):
        if boolean_mask is not None:
            if not boolean_mask[event_i]:
                continue # skip this event's bbox drawing
        # print(dir(events[event_i]))
        # print("logger,timestamp,time,len:", events[event_i].logger, events[event_i].coarse_timestamp, events[event_i].event_time, events[event_i].event_length, )

        # In plot coordinates
        t_start_orig = events[event_i].event_time
        t_end_orig = t_start_orig + events[event_i].event_length
        print(t_start_orig, t_end_orig)

        # In data coordinates
        # t_start = int( events[event_i].event_time*Fs )
        # t_end = t_start + int( events[event_i].event_length*Fs )
        # print(rawAudio[t_start:t_end, : ].shape)

        #tmp_k = (1 + event_i) * (2000 / (len(events) + 1))
        #xy, width, height = (t_start_orig, tmp_k), 4, 100
        xy, width, height = (t_start_orig, 0), 4, 2000-30

        # Draw a diagonal line
        point1 = [xy[0], xy[1]]
        point2 = [xy[0] + width, height]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        line_handler = ax.plot(x_values, y_values, linewidth=2)
        c = line_handler[0].get_color() # 'r'

        # Draw a rectangle around event:
        # xy, width, height
        rect = Rectangle(xy, width, height, linewidth=2, edgecolor=c, facecolor='none')
        ax.add_patch(rect)


        # Line:
        # plt.axvline(x=t_start_orig)
        # plt.axvline(x=t_end_orig)

    ax.set_xlim(orig_xlim)

viz = audiocore.AudioViz()
fig = plt.figure(figsize=(10 * 4, 4 * 4), dpi=60)
viz.plotSpectrogram(rawAudio)

ax_list = fig.axes
draw_bboxes_for_events(events, ax_list[0])
draw_bboxes_for_events(non_events, ax_list[1], nonoverlapping_nonevent_flags)

final_non_events = [non_events[i] for i in range(len(non_events)) if nonoverlapping_nonevent_flags[i]]

print("We got", len(events), "events.")
print("And", len(final_non_events), "non-events.")

#for ax in ax_list:
#    draw_bboxes_for_events(events, ax)
#    draw_bboxes_for_events(non_events, ax)
plt.show()


# plt.figure(figsize=(8, 6), dpi=180)
# viz.plotSpectrolines(rawAudio,spacing=0.05,low_band=10,high_band=200,step=2)
# plt.show()