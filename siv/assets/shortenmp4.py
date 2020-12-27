import imageio

reader = imageio.get_reader( './faceswap.mp4' )
fps = reader.get_meta_data()['fps']

writer = imageio.get_writer( './short_faceswap.mp4', fps=fps )

counter = 0
for im in reader:
    counter += 1
    if counter > 100 and counter < 350:
        writer.append_data( im )

writer.close()


