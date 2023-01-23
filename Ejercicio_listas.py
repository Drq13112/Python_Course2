"""

Escribe un algoritmo  que busque todas las palabras inversas de una lista.
Ejemplo de palabras inversas: radar, oro, rajar, rallar, salas, somos, etc...

Aprtado opcional:
    
Una vez que se han encontrado todas las palabras inversas, las almecenamos en otra lista 
donde se comprobará que no hay palabras repetidas y si las hay se eliminarán todos los duplicados 
de la lista hasta que solo quede una.
"""

# Original array
L = ["radar", 'oro', 'comida', 'perro', 'rajar',
     'rallar', 'paco', 'salas', 'sala', 'somos', 'salas']

L_inverse = []
# Creating the array based only of inverse words

for word in L:

    length = int(len(word)/2)

    if len(word) % 2 == 1:

        # Impar
        word1 = word[0:length+1]
        word2 = word[length:len(word)]

    else:

        # par
        word1 = word[0:length]
        word2 = word[length:len(word)]

    word1_in_letters = list(word1)
    word2_in_letters = list(word2)
    Aux_word = list()

    # comparing the words
    for i in reversed(word2_in_letters):

        Aux_word.append(i)

    if Aux_word == word1_in_letters:
        L_inverse.append(word)


# Let's compare the words stored in the array

for index, word1 in enumerate(L_inverse):
    for word2 in L_inverse[index+1:len(L_inverse)]:
        if word1 == word2:
            L_inverse.remove(word2)


"""
Leer solo los letras de un fichero de texto que contine tanto letras como numeros
y los escribes en otros 2 ficheros. Uno solo de numeros y otro solo de letras.

Elimina el caracter \n de salto de pagina para que al escribir en el fichero no 
se haga un salto de pagina.

Nota: usa la función .isdigit() para distinguir entre numeros y letras

"""
L_numbers = []
L_letters = []
with open('Exam.txt', 'r') as file:
    for line in file:
        #print(line, end='')
        L = list(line)

        print(L)
        for n in L:

            if True == n.isdigit():
                L_numbers.append(n)
            else:
                if n != "\n":
                    L_letters.append(n)


with open('Examen_numero.txt', 'w') as file:
    for number in L_numbers:
        file.write(number)

with open('Examen_letras.txt', 'w') as file:
    for letter in L_letters:
        file.write(letter)


"""

Ejercico de funciones:
    
    Escribe una función llamada FindMedian().
    If the length of the array is pair, it returns the average value of the two median values.
    Otherwise, ir the length is odd, the function returns the median value.
    Use the function sort to order the numbers stored inside
    
    Example:
        
        [15.0, 5.3, 18.2] it returns 15.0
        [1.0, 2.0, 3.0, 4.0] it returns 2.5, the average of 2.0 and 3.0
        

"""

L = [5.0, 6.0, 7.0, 8.0]


def FindMedian(L):

    if len(L) == 0:
        print("EMPTY LIST")
        
        exit()

    L_copy = L[:]

    L_copy.sort()

    if len(L_copy) % 2 == 1:

        return L_copy[len(L_copy)//2]

    else:
        return (L_copy[len(L_copy)//2]+L_copy[len(L_copy)//2-1])/2


result = FindMedian(L)

print(result)
