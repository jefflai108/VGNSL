SpokenCOCO data pre-processing: 
    1) generate parse tree for text transcriptions --> done 
    2) force alignment with MFA --> done

SpokenCOCO baseline: 
    1) re-run VG-NSL on SpokenCOCO. 


Force Alignment during training: 

    version A: force-aligner + average-pool segment embedding 
        version A.1: 
            spectrograms --> force alignment --> pool over each segment to get segment-level embeddings --> VG-NSL 

        version A.2: 
            pre-trained hubert/wav2vec2 --> frame-level force alignment --> segment embeddings --> VG-NSL 

        version A.3: 
            pre-trained ResDaveNet VQ2/VQ3 --> frame-level force alignment --> run-length encoding --> segment embeddings --> VG-NSL 

    version B: force-aligner only
        version A.1: 
            spectrograms --> force alignment --> VG-NSL 

        version A.2: 
            pre-trained hubert/wav2vec2 --> frame-level force alignment --> VG-NSL 

        version A.3: 
            pre-trained ResDaveNet VQ2/VQ3 --> frame-level force alignment --> run-length encoding --> VG-NSL 

Force Alignment during testing: 

    ground truth text 
