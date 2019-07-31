""" Class to Import Chunked Data """


class ImportChunkData:
    def __init__(self, bird_id, sess_name, location=None):
        assert isinstance(bird_id, str)
        assert isinstance(sess_name, str)

        # Initialize Session Identifiers
        self.bird_id = bird_id
        self.date = sess_name