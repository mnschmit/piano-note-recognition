import midi
import numpy as np

def midi_matrix(filename, min_pitch=36):
    pattern = midi.read_midifile(filename)

    resolution = pattern.resolution
    tempo = 120 # arbitrary default bpm

    max_tick = max([sum(map(lambda e: e.tick, track)) for track in pattern])

    matrix = np.zeros((84, max_tick))
    for track in pattern:
        last_event_when = 0 # in ticks
        currently_played_notes = set()
        for event in track:
            if isinstance(event, midi.SetTempoEvent):
                tempo = event.get_bpm()
                continue

            if isinstance(event, midi.NoteOnEvent) or isinstance(event, midi.NoteOffEvent):
                # what happened in-between
                for note in currently_played_notes:
                    for tick in range(last_event_when, last_event_when + event.tick):
                        matrix[note - min_pitch][tick] = 1

                last_event_when += event.tick

                # the case of a noteOffEvent
                if event.get_velocity() == 0:
                    currently_played_notes.discard(event.get_pitch())
                else:
                    # the case of a noteOnEvent
                    currently_played_notes.add(event.get_pitch())

    return matrix


### DO NOT USE - IT'S BUGGY AND USELESS AND DEPRECATED ###
def midi_to_matrix(filename, time_res, matrix, min_pitch=36):
    pattern = midi.read_midifile(filename)

    resolution = pattern.resolution
    tempo = 120 # default bpm (just an arbitrary guess)


    ## an often encountered structure in the sample data is the following:
    ## first track: only metadata such as key and tempo
    ## second track: one and only track with actual notes
    for track in pattern:
        last_event_when = 0 # in ticks
        last_bin = 0
        currently_played_notes = set()
        for event in track:
            if isinstance(event, midi.SetTempoEvent):
                tempo = event.get_bpm()
                continue

            if isinstance(event, midi.NoteOnEvent) or isinstance(event, midi.NoteOffEvent):
                last_event_when += event.tick
                when = last_event_when * (60.0 / tempo / resolution) # in seconds
                #when /= 3 # arbitrary value to fit better expectations
                corresponding_bin = int(when / time_res)
                
                # what happened in-between
                for note in currently_played_notes:
                    for bin in range(last_bin, corresponding_bin):
                        #matrix[note - min_pitch][bin] = 1
                        pass

                last_bin = corresponding_bin

                # the case of a noteOffEvent
                if event.get_velocity() == 0:
                    currently_played_notes.discard(event.get_pitch())
                else:
                    # the case of a noteOnEvent
                    currently_played_notes.add(event.get_pitch())

    return matrix
