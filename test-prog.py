    
    
    
    # Tooltip Annotation 
    def hover_annotate(event):
         if ax_save_button.contains(event)[0]:
             pass
    fig.canvas.mpl_connect('motion_notify_event', hover_annotate)  