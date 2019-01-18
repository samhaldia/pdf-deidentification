from django.db import models


# Create your models here.

class Document(models.Model):
    # Model for Uploading Documents
    docfile = models.FileField(upload_to='documents')
	
    def __str__(self):
        """A string representation of the model."""
        return self.docfile.name

    def delete(self, *args, **kwargs):
        # object is being removed from db, remove the file from storage first
        self.docfile.delete()
        return super(Document, self).delete(*args, **kwargs)
	
	
class ProcDocument(models.Model):
    # Model For Processed Document
    procdocfile = models.FileField(upload_to='procdocuments')
	
    def __str__(self):
        """A string representation of the model."""
        return self.procdocfile.name
		
    def delete(self, *args, **kwargs):
        # object is being removed from db, remove the file from storage first
        self.procdocfile.delete()
        return super(ProcDocument, self).delete(*args, **kwargs)
