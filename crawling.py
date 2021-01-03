from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

arguments = {"keywords":"잘생긴 연예인","limit":100,"print_urls":True,"format":"jpg"}
paths = response.download(arguments)
print(paths) 