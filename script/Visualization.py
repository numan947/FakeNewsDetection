import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from sklearn import metrics 

GEN_DIR = "../generated/"
FIG_DIR = "../generated/figures/"


def create_count_plot(df:pd.DataFrame, x:str, title:str=None, xlabel:str=None, ylabel:str=None, filename:str=None):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=x, order=df[x].value_counts().index)
    
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if filename:
        plt.savefig(
            FIG_DIR+filename+".pdf", 
            bbox_inches='tight', 
            pad_inches=0
            )
    plt.close()


def draw_wordcloud(wordlist, save_name):
    wordCloud = WordCloud(width=1600, 
                      height=800, 
                      random_state=43, 
                      max_font_size=110, 
                      collocations=False,
                        background_color='white'
                         ) 
    plt.figure(figsize=(15, 10)) 
    plt.imshow(wordCloud.generate(wordlist), interpolation='bilinear') 
    plt.axis('off') 
    plt.savefig(
        FIG_DIR+save_name+"_wordcloud.pdf", 
        bbox_inches='tight', 
        transparent=True,
        pad_inches=0
    )


def word_clouds_for_liar_data(data, save_name="liar_data"):
    consolidated = ' '.join( word for word in data['stmt'][data['label'] == 'false'].astype(str)) 
    draw_wordcloud(consolidated, save_name+"_false")
    consolidated = ' '.join( word for word in data['stmt'][data['label'] == 'half-true'].astype(str)) 
    draw_wordcloud(consolidated, save_name+"_half-true")
    consolidated = ' '.join( word for word in data['stmt'][data['label'] == 'mostly-true'].astype(str)) 
    draw_wordcloud(consolidated, save_name+"_mostly-true")
    consolidated = ' '.join( word for word in data['stmt'][data['label'] == 'true'].astype(str)) 
    draw_wordcloud(consolidated, save_name+"_true")
    consolidated = ' '.join( word for word in data['stmt'][data['label'] == 'barely-true'].astype(str)) 
    draw_wordcloud(consolidated, save_name+"_barely-true")
    consolidated = ' '.join( word for word in data['stmt'][data['label'] == 'pants-fire'].astype(str)) 
    draw_wordcloud(consolidated, save_name+"_pants-fire")


def create_confusion_matrix(confusion_matrix, labels, title:str=None, filename:str=None):
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, 
        display_labels=labels
        ) 
  
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size
    cm_display.plot(ax=ax, cmap='Blues', xticks_rotation=45)  # Choose color map and rotate xticks

    # Add title and axis labels
    if title:
        plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Add grid lines for better readability
    plt.grid(False)
    
    if filename:
        plt.savefig(
            FIG_DIR+filename+"_confusion_matrix.pdf", 
            bbox_inches='tight', 
            pad_inches=0
            )
    plt.close()

