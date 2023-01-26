import sys


def get_identation_level(line: str) -> int:
    identation_level = 0
    for char in line:
        if char != ' ':
            break
        identation_level += 1
    return identation_level

if __name__ == "__main__":
    # 1- create an array of lines from the input
    text = sys.stdin.read().split('\n')
    last_identation_level = current_identation_level = 0
    for i in range(len(text)):
        # 2- count how many trailing spaces the line has in its beginning
        current_identation_level = get_identation_level(text[i])
        # if the line trailing spaces count is different from the last one, add a "\n$$$$$$" in the end of the line right before it
        if current_identation_level != last_identation_level:
            text[i-1] += "\n$$$$$$$"
        last_identation_level = current_identation_level
    # 3- recreate the output by joining the lines in a single string
    text = '\n'.join(text)
    print(text)
