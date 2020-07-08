def getdata( fname ):
    fle = open ( fname , "r" )
    str = fle.read().splitlines()
    str = str[1:]
    data = [item.split(',') for item in str]
    data = [[item[0],float(item[1])] for item in data ]
    ret = [[],[]]
    for item in data:
        ret[0].append ( item[0] )
        ret[1].append ( item[1] )
    #print ( data )
    return ret

def getdata_nolabel( fname ):
    fle = open ( fname , "r" )
    str = fle.read().splitlines()
    str = str[1:]
    data = [item.split(',') for item in str]
    data = [[item[0]] for item in data ]
    ret = [[],[]]
    for item in data:
        ret[0].append ( item[0] )
        ret[1].append ( 0 )
    #print ( data )
    return ret