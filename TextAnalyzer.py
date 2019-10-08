import sys
import argparse
import numpy as np
from pyspark import SparkContext

def toLowerCase(s):
    """ Convert a string to lowercase. E.g., 'BaNaNa' becomes 'banana'
    """
    return s.lower()

def stripNonAlpha(s):
    """ Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """
    return ''.join([c for c in s if c.isalpha()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Text Analysis through TFIDF computation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('mode', help='Mode of operation',choices=['TF','IDF','TFIDF','SIM','TOP']) 
    parser.add_argument('input', help='Input file or list of files.')
    parser.add_argument('output', help='File in which output is stored')
    parser.add_argument('--master',default="local[20]",help="Spark Master")
    parser.add_argument('--idfvalues',type=str,default="idf", help='File/directory containing IDF values. Used in TFIDF mode to compute TFIDF')
    parser.add_argument('--other',type=str,help = 'Score to which input score is to be compared. Used in SIM mode')
    args = parser.parse_args()
  
    sc = SparkContext(args.master, 'Text Analysis')

    if args.mode=='TF' :
        # Read text file at args.input, compute TF of each term, 
        # and store result in file args.output. All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings, i.e., "" 
        # are also removed
        lines=sc.textFile(args.input)
        res=lines.flatMap(lambda line: line.split(" "))\
		.map(toLowerCase)\
		.map(stripNonAlpha)\
		.map(lambda word: (word,1))\
                .filter(lambda x:len(x[0])!=0)\
		.reduceByKey(lambda x, y: x+y)\
                .saveAsTextFile(args.output)
    if args.mode=='TOP':
        # Read file at args.input, comprizing strings representing pairs of the form (TERM,VAL), 
        # where TERM is a string and VAL is a numeric value. Find the pairs with the top 20 values,
        # and store result in args.output
        
        tfscore=sc.textFile(args.input)
        
        res=lines.flatMap(lambda line:line.split(" "))\
                .map(toLowerCase)\
                .map(stripNonAlpha)\
                .map(lambda word: (word,1))\
                .filter(lambda x: len(x[0])!=0)\
                .reduceByKey(lambda x,y:x+y)\
                .sortBy(lambda x: x[1],False).take(20)
        #print(res)
        new_rdd=sc.parallelize([res]).saveAsTextFile(args.output)
       	

    if  args.mode=='IDF':
        # Read list of files from args.input, compute IDF of each term,
        # and store result in file args.output.  All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings ""
        # are removed
        pairs=sc.wholeTextFiles(args.input)
        cnt=pairs.count()
        word_num=pairs.flatMapValues(lambda pair: pair.split(" "))\
                      .mapValues(toLowerCase)\
                      .mapValues(stripNonAlpha)\
                      .filter(lambda x:len(x[1])!=0)\
                      .distinct()\
                      .values().map(lambda word:(word,1))\
                      .reduceByKey(lambda x,y: x+y)
        word_num.mapValues(lambda v:np.log(1.*cnt/(v+1)))\
                .saveAsTextFile(args.output)

    if args.mode=='TFIDF':
        # Read  TF scores from file args.input the IDF scores from file args.idfvalues,
        # compute TFIDF score, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL),
        # where TERM is a lowercase letter-only string and VAL is a numeric value. 
        
        tf_score=sc.textFile(args.input).map(lambda x:eval(x))
        idf_score=sc.textFile(args.idfvalues).map(lambda x:eval(x))
        temp=tf_score.join(idf_score)
        res=temp.mapValues(lambda x: x[0]*x[1]).saveAsTextFile(args.output)
        #.sortBy(lambda x: x[1],False).take(20)
        #print(res)
    if args.mode=='SIM':
        # Read  scores from file args.input the scores from file args.other,
        # compute the cosine similarity between them, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL), 
        # where TERM is a lowercase, letter-only string and VAL is a numeric value. 
        w1=sc.textFile(args.input)\
             .map(lambda x :eval(x))
        w2=sc.textFile(args.other)\
             .map(lambda x:eval(x))
        numer=w1.join(w2)\
                .mapValues(lambda x:x[0]*x[1])\
                .reduce(lambda x,y:('ans',x[1]+y[1]))
        numer=numer[1]
        
        deno1=w1.mapValues(lambda x: x**2).reduce(lambda x,y:('ans',x[1]+y[1]))
        deno2=w2.mapValues(lambda x: x**2).reduce(lambda x,y:('ans',x[1]+y[1]))
        
        deno=deno1[1]*deno2[1]
        deno=np.sqrt(deno)
        
        f=open(args.output,'w')
        f.write(str(numer/deno))
        f.write('\n')
        f.close()


                
