def rdbms_main():
  n = 0
	try:
        response = get_final_output_from_model()
	except Exception as e:
        time.sleep(8)  # sleep for 8 seconds
        while n < 5:
           	try:
               	response = get_final_output_from_model()
           	except Exception as e:
               	n += 1
               	print(
                  		"error calling open AI, I am retrying 5 attempts , attempt ", n
               	)
               	time.sleep(8)  # sleep for 8 seconds
               	print(e)
	
	display_output(response)
