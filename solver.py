from collections import defaultdict


class Cell(object):
    remaining_values = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    final_value = None

    def __init__(self, row, column, board, initial_value=0):
        self.row = row
        self.column = column
        self.board = board

        if initial_value:
            self.final_value = initial_value
            self.remaining_values = set()

    def __repr__(self):
        if self.final_value:
            return str(self.final_value)
        else:
            return str(sorted(list(self.remaining_values)))

    def _cells_in_row(self):
        for cell in self.board[self.row]:
            yield cell

    def _cells_in_column(self):
        for row in self.board:
            yield row[self.column]

    def _cells_in_square(self):
        indices = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
        row_indices = indices[self.row / 3]
        column_indices = indices[self.column / 3]

        for r in row_indices:
            for c in column_indices:
                yield self.board[r][c]

    def process(self):
        self._update_possibilities()
        self._check_loneliness()
        if not self.final_value and len(self.remaining_values) == 1:
            self.final_value = self.remaining_values.pop()

    def _update_possibilities(self):
        if self.final_value:
            return

        external_final_values = set(
            [] # just for formatting, lol
        ).union(
            set([x.final_value for x in self._cells_in_row() if x.final_value])
        ).union(
            set([x.final_value for x in self._cells_in_column() if x.final_value])
        ).union(
            set([x.final_value for x in self._cells_in_square() if x.final_value])
        )

        self.remaining_values = self.remaining_values - external_final_values

        if len(self.remaining_values) == 1:
            self.final_value = self.remaining_values.pop()

    def _get_possibility_counts(self, cells):
        counter = defaultdict(int)
        for cell in cells:
            for value in cell.remaining_values:
                counter[value] += 1
        return counter

    def _check_loneliness(self):
        row_counts = self._get_possibility_counts(self._cells_in_row())
        column_counts = self._get_possibility_counts(self._cells_in_column())
        square_counts = self._get_possibility_counts(self._cells_in_square())

        for remaining_value in self.remaining_values:
            if (
                    (row_counts[remaining_value] == 1) or
                    (column_counts[remaining_value] == 1) or
                    (square_counts[remaining_value] == 1)
            ):
                self.final_value = remaining_value
                self.remaining_values = set()
                return


class Board(object):

    def __init__(self, board):
        # initialise rows to put cell objects in
        self._cell_board = [[], [], [], [], [], [], [], [], []]

        for row in xrange(9):
            for column in xrange(9):
                value = board[row][column]
                cell = Cell(row, column, self._cell_board, value)
                self[row].append(cell)

    @property
    def reached_contradiction(self):
        return any(
            (not cell.final_value and not cell.remaining_values)
            for cell in self.cells
        )

    @property
    def solved(self):
        return all(cell.final_value for cell in self.cells)

    @property
    def cells(self):
        for row in xrange(9):
            for column in xrange(9):
                cell = self[row][column]
                yield cell

    @property
    def basic_representation(self):
        rows = []

        for row in self._cell_board:
            rows.append([cell.final_value or 0 for cell in row])

        return rows

    def __getitem__(self, i):
        return self._cell_board[i]

    def __repr__(self):
        line_format   = "{} {} {} | {} {} {} | {} {} {}"
        row_separator = "------+-------+------"
        number_display = lambda x: x.final_value if x.final_value else ' '
        lines = []

        lines.append(line_format.format(*map(number_display, self[0][0]._cells_in_row())))
        lines.append(line_format.format(*map(number_display, self[1][0]._cells_in_row())))
        lines.append(line_format.format(*map(number_display, self[2][0]._cells_in_row())))
        lines.append(row_separator)
        lines.append(line_format.format(*map(number_display, self[3][0]._cells_in_row())))
        lines.append(line_format.format(*map(number_display, self[4][0]._cells_in_row())))
        lines.append(line_format.format(*map(number_display, self[5][0]._cells_in_row())))
        lines.append(row_separator)
        lines.append(line_format.format(*map(number_display, self[6][0]._cells_in_row())))
        lines.append(line_format.format(*map(number_display, self[7][0]._cells_in_row())))
        lines.append(line_format.format(*map(number_display, self[8][0]._cells_in_row())))
        return '\n' + '\n'.join(lines) + '\n'

    def find_smallest_guess(self):
        # bit of an optimisation - if there's a cell with like 9 things you could guess that,
        # that might take a while. So let's just find the cell with the fewest possibilities
        # and pick one of those.

        row = None
        column = None
        guesses = None

        for cell in self.cells:
            if not cell.remaining_values:
                continue

            if (guesses is None) or len(guesses) > len(cell.remaining_values):
                row = cell.row
                column = cell.column
                guesses = cell.remaining_values

        if not guesses:
            print("wtf")
            raise AssertionError

        return row, column, guesses


    @property
    def is_valid(self):
        # well this is an overly-onerous way of doing this, but screw it.
        for cell in self.cells:
            row_values = set(x.final_value for x in cell._cells_in_row())
            column_values = set(x.final_value for x in cell._cells_in_column())
            square_values = set(x.final_value for x in cell._cells_in_square())

            all_equal = row_values == column_values == square_values == {1, 2, 3, 4, 5, 6, 7, 8, 9}
            if not all_equal:
                return False
        return True

    @property
    def state(self):
        # gives a string representation of the full state of the board
        # including assumptions
        return str(list(self.cells))

    def solve(self, verbose=True, recursion_level=0):
        if recursion_level > 5:
            raise Exception
        while True:
            if verbose:
                print(self)

            state_0 = self.state

            for cell in self.cells:
                cell.process()

            state_1 = self.state

            if state_0 == state_1:
                if self.solved:
                    if self.is_valid:
                        return self

                    return False

                if self.reached_contradiction:
                    if verbose:
                        print("contradiction found.")
                    return False

                # stalemate, then - need to make a guess.
                row, column, guesses  = self.find_smallest_guess()
                for guess in guesses:
                    basic_representation = self.basic_representation
                    basic_representation[row][column] = guess

                    new_board = Board(basic_representation)
                    if verbose:
                        print("Making assumption: row={row}, col={col}, value={guess}...".format(row=row, col=column, guess=guess))
                        print(self)
                    result = new_board.solve(verbose=verbose, recursion_level=recursion_level+1)
                    if result:
                        return result
                return False


test_board_1 = [
    [7, 5, 0, 0, 3, 8, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 9, 0, 0],
    [0, 9, 0, 0, 7, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 8, 2, 1],
    [1, 3, 4, 0, 0, 0, 7, 5, 6],
    [2, 8, 7, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 0, 2, 0, 0, 3, 0],
    [0, 0, 5, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 4, 1, 0, 0, 9, 2],
]

test_board_2 = [
    [0, 0, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 8, 0, 6, 0, 0, 0, 0],
    [7, 0, 5, 4, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 3, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 5, 0],
    [5, 8, 7, 0, 0, 0, 2, 0, 0],
    [4, 0, 0, 7, 0, 0, 9, 6, 0],
    [9, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 2, 6, 5, 0, 0, 4, 0, 0],
]

test_board_3 = [
    [0, 0, 0, 5, 0, 4, 0, 0, 0],
    [0, 0, 8, 0, 0, 0, 3, 0, 0],
    [7, 5, 0, 0, 2, 0, 0, 0, 6],
    [0, 0, 0, 7, 8, 0, 2, 6, 4],
    [0, 6, 0, 0, 9, 0, 0, 0, 8],
    [5, 0, 0, 0, 0, 6, 0, 0, 0],
    [0, 0, 5, 0, 3, 0, 9, 0, 7],
    [0, 0, 2, 6, 0, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 3, 0],
]


# The hardest button to button
test_board_x = [
    [8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 6, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 9, 0, 2, 0, 0],
    [0, 5, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 4, 5, 7, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 3, 0],
    [0, 0, 1, 0, 0, 0, 0, 6, 8],
    [0, 0, 8, 5, 0, 0, 0, 1, 0],
    [0, 9, 0, 0, 0, 0, 4, 0, 0],
]

#board = Board(test_board_x)
#board.solve()
