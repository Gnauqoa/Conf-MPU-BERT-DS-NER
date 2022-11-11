with open('ori_test.txt', 'r', encoding='utf-8') as ALL, open('test.txt', 'w', encoding='utf-8') as Fully:
    for line in ALL.readlines():
        if len(line.strip()) > 0:
            line_info = line.strip().split(' ')
            if 'PER' in line_info[1]:
                Fully.writelines(line_info[0] + ' ' + line_info[1] + '\n')
            elif 'LOC' in line_info[1]:
                Fully.writelines(line_info[0] + ' ' + line_info[1] + '\n')
            elif 'ORG' in line_info[1]:
                Fully.writelines(line_info[0] + ' ' + line_info[1] + '\n')
            elif 'MISC' in line_info[1]:
                Fully.writelines(line_info[0] + ' ' + line_info[1] + '\n')
            else:
                Fully.writelines(line_info[0] + ' ' + line_info[1] + '\n')
        else:
            Fully.writelines(line)
