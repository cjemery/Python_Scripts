#!/usr/bin/env python

import os,sys

def PrintMenu():
	print("Option1")
	print("Option2")
	print("Option3")
	print("Option4")

def Main():

	if selection == 1:
		print("you selected Option 1\n")

	elif selection == 2:
		print("you selected Option 2\n")

	elif selection == 3:
		print("you selected Option 3\n")

	elif selection == 4:
		print("you selected Option 4\n")

while (True):
	print("Select 1-4 from the menu below. Press 0 to exit: ")
	#print('\n')

	PrintMenu()

	selection = input('') 
	if selection == 0:
		break;
	Main()







