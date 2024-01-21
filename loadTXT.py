import os


# 生成train_data.txt用于训练
def loadTXT(config):
    listTXT = open("train_data.txt", "w")
    total = 0
    root = config.locate
    for idx, name in enumerate(config.name):
        path = root + '/' + name
        files = os.listdir(path)
        files = sorted(files)
        for i, file in enumerate(files):
            pPath = os.path.join(path, file)
            if not os.path.isdir(pPath):
                continue
            pNames = os.listdir(pPath)
            if len(pNames) < config.minNum:
                continue
            total = total + 1
            count = 0
            for pName in pNames:
                if pName.endswith(config.type[idx]) and count < config.maxNum:
                    count = count + 1
                    listTXT.write(str(total) + ";" + '%s' % (os.path.join(os.path.abspath(path), file, pName)))
                    listTXT.write('\n')
                if count >= config.maxNum:
                    break
    listTXT.close()
    return
