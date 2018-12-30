from submission import *

res_dict = {}
turn_dict = {}
players = [OriginalReflexAgent, ReflexAgent, MinimaxAgent, AlphaBetaAgent, RandomExpectimaxAgent]
depths = [2, 3, 4]
layouts = ['capsuleClassic', 'contestClassic', 'mediumClassic',
           'minimaxClassic', 'openClassic', 'originalClassic',
           'smallClassic', 'testClassic', 'trappedClassic', 'trickyClassic']

with open('results.csv', 'r') as file_ptr:
    data = file_ptr.readlines()
    for line in data:
        if line in ['\n']:
            continue
        line_items = line.split(',')
        if (line_items[0], line_items[1]) in res_dict:
            res_dict[(line_items[0], line_items[1])] += float(line_items[3])
        else:
            res_dict[(line_items[0], line_items[1])] = float(line_items[3])

        if (line_items[0], line_items[1]) in turn_dict:
            turn_dict[(line_items[0], line_items[1])] += float(line_items[4])
        else:
            turn_dict[(line_items[0], line_items[1])] = float(line_items[4])
    f = open('graphRes.csv', 'w+')
    f.write('Score as a function of depth \n')
    for (x, y) in res_dict:
        avg_score = res_dict[(x, y)] / 10
        f.write(str(x) + "," + str(y) + "," + str(avg_score))
        f.write('\n')
    f.write('\n')
    f.write('runtime as a function of depth \n')
    for (x, y) in turn_dict:
        avg_turn = turn_dict[(x, y)] / 10
        f.write(str(x) + "," + str(y) + "," + str(avg_turn))
        f.write('\n')
