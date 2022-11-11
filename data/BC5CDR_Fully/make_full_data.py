with open('ori_valid.txt', 'r', encoding='utf-8') as ALL, open('valid', 'w', encoding='utf-8') as Fully:
    for line in ALL.readlines():
        if len(line.strip()) > 0:
            line_info = line.strip().split(' ')
            if 'Chemical' in line_info[1]:
                Fully.writelines(line_info[0] + ' ' + line_info[1] + '\n')
            elif 'Disease' in line_info[1]:
                Fully.writelines(line_info[0] + ' ' + line_info[1] + '\n')
            else:
                Fully.writelines(line_info[0] + ' ' + line_info[1] + '\n')
        else:
            Fully.writelines(line)
