

def read_file(fname):
    data = dict()
    with open(fname) as file:
        file.readline()
        for line in file:
            line = line.split(",")
            userid = int(line[2])
            if userid in data.keys():
                data[userid] += 1
            else:
                data[userid] = 1
    return data
    
def histogram(data):
    binning = dict()
    for value in data.values():
        if value in binning.keys():
            binning[value] += 1
        else:
            binning[value] = 1
    return binning
    
hist = histogram(read_file('train.csv'))

totusers = 0
wfreq = 0
m = max(hist.keys())
for x in xrange(m+1):
    if x in hist.keys():
        v = hist[x]
    else:
        v = 0
    print x, v
    totusers += v
    wfreq += x*v
    
#print "Mean is ", wfreq/float(totusers)
    
