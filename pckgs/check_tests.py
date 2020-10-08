import json 
import requests 
import pandas as pd 
import os
import matplotlib.pyplot as plt 

# 'https://tests.yt-project.org/job/yt_py38_git/1552/testReport/api/json?pretty=true'
pr_tnums={2931:1552,2934:1568}

class test_result(object):
    def __init__(self,PR,testnum):
        self.filename = f"./data/pr_{PR}_test_{testnum}.json"
        
        if not os.path.isfile(self.filename):
            self.fetch_write_json(PR,testnum)
            
        with open(self.filename,'r') as f:
            self.results = json.load(f)
        
        self.set_df()
        
    def fetch_write_json(self,PR,testnum):
        url = f"https://tests.yt-project.org/job/yt_py38_git/{testnum}/testReport/api/json"
        r = requests.get(url)
        with open(self.filename,'w') as f: 
            f.write(json.dumps(r.json()))
                
    def set_df(self):
        rows = []                 
        for suite in self.results['suites']:
            for case in suite['cases']:
                rows.append(case) 
        self.df = pd.DataFrame(rows)
        
def compare_two_tests(PR1,PR2,**kwargs):
    PR_1 = test_result(PR1,pr_tnums[PR1])
    PR_2 = test_result(PR2,pr_tnums[PR2])
    
    cols = ['className','duration']
    df1 = PR_1.df[cols].groupby(by=['className']).sum().reset_index()
    df2 = PR_2.df[cols].groupby(by=['className']).sum().reset_index() 
    df = pd.merge(df1,df2,on=['className'],how='left',suffixes=(f"_{PR1}", f"_{PR2}"))
    df['timediff']=df[f"duration_{PR1}"]-df[f"duration_{PR2}"]
    
    plot = kwargs.pop('plot',True)
    if plot:
        bins = kwargs.pop('bins',100)
        ax = df['timediff'].hist(bins=bins,**kwargs)        
        ax.set_xlabel(f"PR {PR1} - PR {PR2} [s]")
        ax.set_ylabel('N')
        plt.show() 
    return df 
    
if __name__=='__main__':
    
    PR2934 = ct.test_result(2934,pr_tnums[2934])
    PR2931 = ct.test_result(2931,pr_tnums[2931])

        
