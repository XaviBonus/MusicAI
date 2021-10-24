import copy as cp
import numpy as np

def cut_sequence(sequence, time):
    """ Crops a sequence at a given time"""
    ns_cut = cp.deepcopy(sequence)
    for n in sequence.notes:
        if n.start_time > time:
            ns_cut.notes.remove(n)
    ns_cut.total_time = max([n.end_time for n in ns_cut.notes])
    return ns_cut

def encode(sequences, music_vae):
    """Encodes the sequences using the model in music_vae."""
    latent = []
    for s in sequences:
        extracted_tensors = music_vae._config.data_converter.to_tensors(sequences[1])

        inputs = []
        controls = []
        lengths = []

        for i,T in enumerate(extracted_tensors.inputs):
            inputs.append(extracted_tensors.inputs[i])
            controls.append(extracted_tensors.controls[i])
            lengths.append(extracted_tensors.lengths[i])

        latent.append(music_vae.encode_tensors(inputs, lengths, controls))

    return latent

def many_to_one(sequences, music_vae, length = 300, temperature = 1):
    """Averages a list of NotesSquences in latent space, and returns a melody."""
    vectors = encode(sequences, music_vae)
    average_vectors = np.mean(np.mean(vectors,axis=0), axis=1)
    result = music_vae.decode(average_vectors, length = length, temperature = temperature)
    return result[0]
