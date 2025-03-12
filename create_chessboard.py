import numpy as np
import matplotlib.pyplot as plt


def create_chessboard(rows=6, cols=9, square_size=100):
    """
    Creates a chessboard pattern for camera calibration.
    :param rows: Number of squares along the row (excluding the outer border).
    :param cols: Number of squares along the column (excluding the outer border).
    :param square_size: Size of each square in pixels.
    :return: None
    """
    # Create a chessboard pattern
    chessboard = np.zeros((rows * square_size, cols * square_size), dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2 == 0:
                chessboard[row * square_size:(row + 1) * square_size, col * square_size:(col + 1) * square_size] = 255

    # Plot the chessboard
    plt.figure(figsize=(cols, rows))
    plt.imshow(chessboard, cmap='gray', extent=[0, cols * square_size, 0, rows * square_size])
    plt.axis('off')
    plt.gca().invert_yaxis()

    # Save the chessboard pattern
    plt.savefig('chessboard.png', dpi=300, bbox_inches='tight', pad_inches=0)
    print("Chessboard pattern saved as chessboard.png")


# Create and save the chessboard pattern
create_chessboard()
