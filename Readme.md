# Introduction

This is a Basic Web Based tool to De-Identify PDFs Using Django and Python Libraries pdfminer, pdfrw.

# Requirements

This Repository Used Following :

	Conda 4.0.5

	Django 2.1.2

	pdfrw>=0.4

	defusedxml

	pdfminer.six # Read For More Info https://pdfminer-docs.readthedocs.io/pdfminer_index.html

	chardet
			
	django-crispy-forms 
			

# Installation and Configuration

Steps to Setup Environment in Windows:

1. Install Anaconda
2. Create virtualenv for Web Based Tool: 
	conda create --name deidentify python=3  # Create new ENV deidentify with Python 3
	activate deidentify  # Activate the deidentify ENV
	conda list # To See List Of Packages Installed in Current ENV.
3. conda info --envs # List all ENV's created in Conda 
4. pip install django # It will Download latest Django version into de-dentify ENV
5. pip install pdfrw>=0.4
6. pip install defusedxml
7. pip install pdfminer.six
8. pip install chardet
9. pip install django-crispy-forms # 3rd party package to work with Form in Django 
10. Create a Folder of your Choice ex: DeIdentifyTool
11. Clone the Repository inside the Created Folder
12. Create a Mysql Database as configured at de_identify/settings.py file in Project folder in the code repo.
13. Now need to Run : python manage.py makemigrations pdf_deidentify (This is App Specific Name While Creating app in your Django project)
14. python manage.py migrate pdf_deidentify
15. To Look Admin Interface Run: python manage.py createsuperuser
16. Execute to see the Web tool running: python manage.py runserver  # Admin Interface will be available by appending /admin to the base URL
	
	* Here Mysql is Used as DB Engine
	* One can Use any Other Database of once Choice. 
	* To use another DB Engine, need to Configure the Settings.py accordingly.
	* For more refer the Link https://docs.djangoproject.com/en/2.1/ref/settings/#databases
	
Linux Machine:
	
* Steps will Be Same to Build This Web based tool,except the Commands to install Anaconda in Linux Machine and Others if any.
	
	
