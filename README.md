# sentiment-analysis
Introduction - Sentiment analysis
Sentiment analysis is the computational study of opinions, sentiments and subjectivity embedded in user-generated textual document. Usually, the text is classified into positive, negative or neutral sentiments. The most interesting user-generated textual documents are reviews about entities, prod- ucts or services. These use-generated textual documents are of interest to organizations that own or provide the services; supermarkets, movie produces, restaurants, etc. A more fine-grained analysis of the textual document involves identifying various aspects of the main entity in the document and classifying the opinion expressed about the identified aspects. This type of analysis is termed, aspect-based sentiment analysis (ABSA) and it has gained popularity and importance in the last decade.
In this assignment, you will build, experiment and write a report/paper about aspect-based sen- timent analysis model. To a large extent, this will be a guided experimentation leading to a minor self-directed research. You will work in a group of four (4) students to implement an aspect-based sentiment analysis model using the well-known IMDB movie review dataset (see Chollet (2021, Chp. 11)).
Several of the topics underlying this assignment will be discussed in the lectures. However, you should read ahead of the class and be prepared to ask questions in the lectures.
What needs to be done or considered
1. Study Chapter 11 of (Chollet, 2021) provided with this specification. You will conduct the experiments in the chapter to gain familiarity with building a sentiment analysis model using TensorFlow deep learning framework.
1

2. As a result of conducting the experiments in Chapter 11 of (Chollet, 2021) and the references provided with this specification, you will gain a good understanding of the steps involved in developing a basic sentiment analysis model.
3. In a second step, you will extend your understanding of sentiment analysis by implementing an aspect-based sentiment analysis on the same dataset (IMDB reviews) that you used in (1) above. You will notice that this part of the assignment is deliberately made vague. This is to encourage you to read the reference materials provided and also search for more.
4. The method you choose for your aspect-based sentiment analysis must be based on neural network architecture. You can take hint from the methods described in (Chollet, 2021, Chp. 11) and (G ́eron, 2019, Chp. 16, p. 534) for sentiment analysis.
5. Several references have been provided with this specification on aspect-based sentiment analysis that you will find useful.
6. This assignment requires good project management and you are encouraged to distribute tasks among group members and start conducting experiments early.
7. Write a report according to the template provided. You MUST follow the template in setting out your sections. You can have subsections tailored to your presentation style, but the sec- tion headings MUST not be changed. A LaTeX template has been provided along with this specification. Your report MUST not be more than 5 pages of text, figures and tables. This excludes the title page and references.
8. Your report must cite at least 6 sources (peer-reviewed journal or conference papers or books) to support the literature review section. Do not forget to cite the the source of the dataset.
9. Your report must include graphical outputs. However, you need to be judicious in your choice of the plots that you include. Remember that every graphical plot must have a label and informative caption, and must be described in the text of your report. Otherwise you will lose substantial marks.
10. It is possible that you will use jupyter notebook to develop your code. Please note that you cannot submit a notebook file for this assignment. Only a python source code can be submitted (i.e. a .py file).
11. If your source code does not work or emits error messages, your code will not be debugged or fixed. Your report will be marked out of 50% of the total marks for this assignment.
What needs to be submitted
• You will prepare a “zip” or “rar” file containing your report (6-page text plus title and references pages as PDF file) and Python code (named : aspect_based_sentiment.py) file.
• Your code must be executable as a Python version 3.10 (or higher) code and run from command line as:
2

aspect_based_sentiment.py
and write results indicating that your code works (e.g. classification accuracy for your method) to standard output (stdout).
• Submit the “zip” or “rar” via Moodle dropbox provided on or before the deadline. Assessment of group work
Please read carefully.
Group work provides opportunity for students to explore how to deal with group dynamics and learn/gain near-real-world experience. This opportunity is also fraught with danger because a good student may be disadvantaged when working alongside lazy and irresponsible students. This is an extreme situation because most students are largely motivated to learn and pass the subject and on average groups work well.
Nevertheless, the assessment must reflect and reward the efforts of individual contribution to the work (reading papers, writing/implementing algorithms in code, and writing a convincing report). To this end, a marking scheme has been devised to capture individual effort as well as group effort.
const int numInGroup = 4;// the number of members in group
double totalMaxMark; //maximum marks allocated to assignment (e.g. 150)
double groupMark; //mark awarded to group (report and code)
double individualMark[numInGroup];//mark computed and awarded to group members double groupAssignedMark[numInGroup];// given out of 10
// Each group member has a group-assigned mark
double percentOfTotalMarks[numInGroup];
for (int k = 0; k< numInGroup; ++k)
percentOfTotalMarks[k] = 0.1 * individualMark[k];
//final individual mark
for (int k = 0; k< numInGroup; ++k)
individualMark[k] = (percentOfTotalMarks[k] * groupMark)*30.0/totalMaxMark;
To facilitate this assessment scheme, each group must submit a table similar to that shown below indicating the marks (out of 10) awarded by the group to each member.
Group member:       #1    #2    #3    #4
Contribution mark (out of 10)    8.5   10.0  9.0   3.0
The group can decide to award the marks individually and provide the aggregated mark in a table. It is advisable that each group member is assigned a task whose degree of completion and quality can be measured. For example a group member could be assigned the task of reading a number
3

of papers and writing a good quality summary for use in the “related works” section of the report. Another member could be assigned the task of understanding code and producing modification to suit the experiment the group decides to perform. Yet, another group member could be assigned the task of editor, putting together the various sections submitted by other group members. These are just some ideas. Please hold a group discussion to assign tasks.
Example: Let us take the example of a hypothetical group that provided individual effort/contribution scores as shown above. Further assume that the group scored 120 out of a total mark of 150. The individual mark based on the algorithm above and considering that this part of the assignment is worth 30% is:
Mark scored by report and code: 120/150
Group member:
Contribution mark (out of 10)
Individual mark (out of 30)
#1    #2    #3    #4
8.5   10.0  9.0   3.0
20.4  24.0  21.6  7.2
So, it is a good idea that group members that contribute maximum effort are scored 10/10.
References
Chollet, F. (2021). Deep learning with python (2nd ed.). Shelter Island, NY, USA: Manning Publications Co.
G ́eron, A. (2019). Hands-on machine learning with scikit-learn, keras &and tensorflow: Concepts, tools, and techniques to build intelligent systems (2nd ed.). CA, USA: O’Reilly Media, Inc.
4
