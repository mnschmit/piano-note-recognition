import midi

def midiToMatrix(filename, time_res, matrix):
    pattern = midi.read_midifile(filename)

    resolution = pattern.resolution
    
    tempo = 120 # default bpm (just a guess)
    # STRONG ASSUMPTION: the first track contains only metadata
    for event in pattern[0]:
        if isinstance(event, midi.SetTempoEvent):
            tempo = event.get_bpm

    # STRONG ASSUMPTION: we have only ONE track with notes (the 2nd)
    for event in pattern[1]:
        # relative time from the last event
        when = event.tick
        
        # calculate with time resolution the actual when

        # fill it in the matrix
