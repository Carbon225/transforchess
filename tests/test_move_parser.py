import pytest

from transforchess.parser import san2human, human2san


@pytest.mark.parametrize(
    'san,human',
    [
        ('e4', 'pawn to e4'),
        ('Bbxd5=Q#', 'bishop from b takes d5 and promotes to queen and checkmate'),
    ]
)
def test_move_parser(san, human):
    assert san2human(san) == human
    assert human2san(human) == san
