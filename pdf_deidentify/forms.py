from django import forms

# Web Form For Uploading Document
class DocumentForm(forms.Form):
    docfile = forms.FileField(
        label='Select a File',
        help_text='Max. 42 MB'
    )
    userTextList = forms.CharField(widget=forms.Textarea, label='Enter Comma Separated Text ex:Requestorfn Lastname4,Firstname LastName4')