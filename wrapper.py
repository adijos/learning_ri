import random_words as rw
N, k, b = (100, 50, 20)
block_sz = 1
byletter = 1
text_path = './data/english_text/eng.txt'

RIG = rw.RIGrammar(N,k,b)
print 'adding words'
RIG.add_words(text_path=text_path,byletter=byletter)
print 'learning basis'
RIG.learn_basis_over(text_path=text_path, block_sz=block_sz)
print 'finding basis representations'
reps = RIG.find_reps()
