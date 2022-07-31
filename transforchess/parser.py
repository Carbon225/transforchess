import regex as re


SAN_REGEX = re.compile(r'^(?:(?:(?<piece>[RBNKQ])?(?<fromcol>[a-h])?(?<fromrow>[1-8])?(?<takes>x)?(?<tocol>[a-h])(?<torow>[1-8])(?:=(?<promotion>[QRBN]))?)|(?<castling>O-O(?:-O)?))(?:(?<check>\+)|(?<mate>#))?$')
HUMAN_SAN = re.compile(r'^(?:(?<piece>pawn|rook|knight|bishop|queen|king)(?: from (?<from>[a-h][1-8]|[a-h]|[1-8]))? (?<action>to|takes) (?<to>[a-h][1-8])(?: and promotes to (?<promotion>rook|knight|bishop|queen))?|(?:(?<castling>kingside|queenside) castling))(?: and (?<check>checkmate|check))?$')


def san2human(san: str) -> str:
    move = SAN_REGEX.match(san)
    
    out = ''

    if move.group('castling') == 'O-O':
        out += 'kingside castling'
    elif move.group('castling') == 'O-O-O':
        out += 'queenside castling'
    else:
        if move.group('piece') == 'R':
            out += 'rook'
        elif move.group('piece') == 'N':
            out += 'knight'
        elif move.group('piece') == 'B':
            out += 'bishop'
        elif move.group('piece') == 'Q':
            out += 'queen'
        elif move.group('piece') == 'K':
            out += 'king'
        else:
            out += 'pawn'

        if move.group('fromcol') is not None and move.group('fromrow') is not None:
            out += ' from ' + move.group('fromcol') + move.group('fromrow')
        elif move.group('fromcol') is not None:
            out += ' from ' + move.group('fromcol')
        elif move.group('fromrow') is not None:
            out += ' from ' + move.group('fromrow')

        if move.group('takes') is not None:
            out += ' takes'
        else:
            out += ' to'

        if move.group('tocol') is not None and move.group('torow') is not None:
            out += ' ' + move.group('tocol') + move.group('torow')

        if move.group('promotion') == 'Q':
            out += ' and promotes to queen'
        elif move.group('promotion') == 'R':
            out += ' and promotes to rook'
        elif move.group('promotion') == 'B':
            out += ' and promotes to bishop'
        elif move.group('promotion') == 'N':
            out += ' and promotes to knight'

    if move.group('check') is not None:
        out += ' and check'
    elif move.group('mate') is not None:
        out += ' and checkmate'

    return out


def human2san(human: str) -> str:
    move = HUMAN_SAN.match(human)

    out = ''

    if move.group('castling') == 'kingside':
        out += 'O-O'
    elif move.group('castling') == 'queenside':
        out += 'O-O-O'
    else:
        if move.group('piece') == 'pawn':
            pass
        elif move.group('piece') == 'rook':
            out += 'R'
        elif move.group('piece') == 'knight':
            out += 'N'
        elif move.group('piece') == 'bishop':
            out += 'B'
        elif move.group('piece') == 'queen':
            out += 'Q'
        elif move.group('piece') == 'king':
            out += 'K'
        else:
            raise ValueError('Invalid piece')

        if move.group('from') is not None:
            out += move.group('from')

        if move.group('action') == 'to':
            pass
        elif move.group('action') == 'takes':
            out += 'x'
        else:
            raise ValueError('Invalid action')

        out += move.group('to')

        if move.group('promotion') == 'rook':
            out += '=R'
        elif move.group('promotion') == 'knight':
            out += '=N'
        elif move.group('promotion') == 'bishop':
            out += '=B'
        elif move.group('promotion') == 'queen':
            out += '=Q'

    if move.group('check') == 'check':
        out += '+'
    elif move.group('check') == 'checkmate':
        out += '#'

    return out
