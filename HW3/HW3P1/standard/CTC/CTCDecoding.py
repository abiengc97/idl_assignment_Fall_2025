import numpy as np


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        for i in range(y_probs.shape[1]):
            path_prob *= np.max(y_probs[:, i, 0])
            index = np.argmax(y_probs[:, i, 0])
            if index != 0:
                if blank:
                    decoded_path.append(self.symbol_set[index - 1])
                    blank = 0
                else:
                    if (
                        len(decoded_path) == 0
                        or decoded_path[-1] != self.symbol_set[index - 1]
                    ):
                        decoded_path.append(self.symbol_set[index - 1])
            else:
                blank = 1

        decoded_path = "".join(decoded_path)

        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """

        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
                        batch size for part 1 will remain 1, but if you plan to use your
                        implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """
        seq_len = y_probs.shape[1]
        
        # Initialize: paths ending in blank vs non-blank
        blankPathScore = {"": y_probs[0, 0, 0]}
        pathScore = {}
        for i, symbol in enumerate(self.symbol_set):
            pathScore[symbol] = y_probs[i + 1, 0, 0]
        
        # Iterate through time steps
        for t in range(1, seq_len):
            # Prune: keep only top beam_width paths
            allScores = [(path, score) for path, score in blankPathScore.items()]
            allScores += [(path, score) for path, score in pathScore.items()]
            allScores.sort(key=lambda x: x[1], reverse=True)
            cutoff = allScores[self.beam_width][1] if self.beam_width < len(allScores) else allScores[-1][1]
            
            prunedBlankPaths = {p: s for p, s in blankPathScore.items() if s > cutoff}
            prunedSymbolPaths = {p: s for p, s in pathScore.items() if s > cutoff}
            
            # Extend with blank
            newBlankPathScore = {}
            for path, score in prunedBlankPaths.items():
                newBlankPathScore[path] = score * y_probs[0, t, 0]
            for path, score in prunedSymbolPaths.items():
                if path in newBlankPathScore:
                    newBlankPathScore[path] += score * y_probs[0, t, 0]
                else:
                    newBlankPathScore[path] = score * y_probs[0, t, 0]
            
            # Extend with symbols
            newPathScore = {}
            for path, score in prunedBlankPaths.items():
                for i, symbol in enumerate(self.symbol_set):
                    newPath = path + symbol
                    newScore = score * y_probs[i + 1, t, 0]
                    if newPath in newPathScore:
                        newPathScore[newPath] += newScore
                    else:
                        newPathScore[newPath] = newScore
            
            for path, score in prunedSymbolPaths.items():
                for i, symbol in enumerate(self.symbol_set):
                    # If same as last character, stay in same path
                    if symbol == path[-1]:
                        newScore = score * y_probs[i + 1, t, 0]
                        if path in newPathScore:
                            newPathScore[path] += newScore
                        else:
                            newPathScore[path] = newScore
                    else:
                        # Different symbol, extend path
                        newPath = path + symbol
                        newScore = score * y_probs[i + 1, t, 0]
                        if newPath in newPathScore:
                            newPathScore[newPath] += newScore
                        else:
                            newPathScore[newPath] = newScore
            
            blankPathScore = newBlankPathScore
            pathScore = newPathScore
        
        # Merge paths ending in blank and non-blank
        mergedPathScores = pathScore.copy()
        for path, score in blankPathScore.items():
            if path in mergedPathScores:
                mergedPathScores[path] += score
            else:
                mergedPathScores[path] = score
        
        bestPath = max(mergedPathScores, key=mergedPathScores.get)
        
        return bestPath, mergedPathScores
