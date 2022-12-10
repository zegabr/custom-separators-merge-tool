import sys


def get_identation_level(line: str) -> int:
    identation_level = 0
    for char in line:
        if char != ' ':
            break
        identation_level += 1
    return identation_level

if __name__ == "__main__":
    # 1- pegar input padrao passado, splitar por linha
    text = sys.stdin.read().split('\n')
    last_identation_level = current_identation_level = 0
    for i in range(len(text)):
        # 2- contar espacos de identacao por linha
        current_identation_level = get_identation_level(text[i])
        # a cada linha cuja identacao eh diferente da anterior, adicionar \n$$$$$$ na linha anterior
        if current_identation_level != last_identation_level:
            text[i-1] += "\n$$$$$$$"
        last_identation_level = current_identation_level
    # 3- reprintar o input modificado
    text = '\n'.join(text)
    print(text)
