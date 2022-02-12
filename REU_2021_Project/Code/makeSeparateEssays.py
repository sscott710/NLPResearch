'''last date modified: 6/10/2021
author: Savannah Scott
Purpose: separate The Federalist Paper file from Project Gutenberg into individual 85 essays and save them as txt files
note: can probably be done easier; look at NLTK book for suggestions
edit 8/6/2021:
    works well; does not need to be  run again'''
def text():
    '''open raw text file from Project Gutenberg'''
    file = open('/mnt/linuxlab/home/r21sscott/REU_Project/The Federalist Papers from ProjGuten.txt', 'r') #file path depends on user
    txt = file.readlines()
    file.close()
    
    '''find where book end'''
    for line in txt:
        if line == 'End of the Project Gutenberg EBook of The Federalist Papers, by\n':
            bookEnd = txt.index(line)
            
    '''find where essays start'''
    for line in txt:
        if line == 'FEDERALIST No. 1\n':
            bookStart = txt.index(line)
            
    '''cut intro, title, and Gutenberg text so only text of essays remains; save this in case needed for future use'''
    justEssays = txt[bookStart:bookEnd]
    je = ''
    for line in justEssays:
        je = je + line
    jefile = open('/mnt/linuxlab/home/r21sscott/REU_Project/FedPapersOnlyEssays.txt', 'w')
    jefile.write(str(je))
    jefile.close()
    
    '''divide text into list of individual essays'''
    essays = []
    ess = ''
    n = 1
    for line in justEssays:
        if 'FEDERALIST No. '+str(n) in line:
            ess = ess + line
        elif 'FEDERALIST No. '+str(n+1) in line:
            essays.append(ess)
            n = n +1
            ess = ''
            ess = ess + line
        elif justEssays.index(line) == 19351:#last line
            ess = ess + line
            essays.append(ess)
        else:
            ess = ess + line
            
    '''write to make new file for each essay'''
    for x in range(len(essays)):
        if x < 9:
            e = essays[x]
            name = e[0:16]+'.txt'
            efile = open('/mnt/linuxlab/home/r21sscott/REU_Project/'+name, 'w')
            efile.write(str(e))
            efile.close()
        else:
            e = essays[x]
            name = e[0:17]+'.txt'
            efile = open('/mnt/linuxlab/home/r21sscott/REU_Project/'+name, 'w')
            efile.write(str(e))
            efile.close()

text()
