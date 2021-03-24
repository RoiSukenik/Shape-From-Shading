# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
tensor_image = tensor([[[0.9804, 0.9804, 0.9804],
         [0.9451, 0.9451, 0.9451],
         [1.0000, 1.0000, 0.9922],
         ...,
         [0.9725, 0.9804, 0.9765],
         [0.9843, 1.0000, 0.9882],
         [0.9961, 1.0000, 1.0000]],

        [[1.0000, 1.0000, 1.0000],
         [1.0000, 1.0000, 1.0000],
         [1.0000, 1.0000, 0.9922],
         ...,
         [0.9961, 1.0000, 1.0000],
         [0.9961, 1.0000, 1.0000],
         [0.9765, 0.9922, 0.9804]],

        [[0.9529, 0.9529, 0.9529],
         [1.0000, 1.0000, 1.0000],
         [0.9804, 0.9804, 0.9725],
         ...,
         [0.9961, 1.0000, 1.0000],
         [0.9961, 1.0000, 1.0000],
         [0.9804, 0.9961, 0.9843]],

        ...,

        [[1.0000, 0.9961, 0.9843],
         [0.9961, 0.9882, 0.9765],
         [1.0000, 1.0000, 0.9843],
         ...,
         [0.9843, 0.9922, 0.9961],
         [0.9686, 0.9765, 0.9882],
         [0.9882, 0.9961, 1.0000]],

        [[0.9569, 0.9490, 0.9373],
         [0.9804, 0.9725, 0.9608],
         [1.0000, 0.9961, 0.9843],
         ...,
         [1.0000, 1.0000, 1.0000],
         [0.9961, 1.0000, 1.0000],
         [0.9961, 1.0000, 1.0000]],

        [[1.0000, 0.9922, 0.9804],
         [1.0000, 0.9961, 0.9843],
         [0.9725, 0.9647, 0.9529],
         ...,
         [1.0000, 1.0000, 1.0000],
         [0.9961, 1.0000, 1.0000],
         [0.9882, 0.9961, 1.0000]]])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plt.imshow(  tensor_image.permute(1, 2, 0)  )
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
