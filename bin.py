
                    # det = detect_book(frame)
                    # if det is not None:
                    #     class_id = det[1]
                    #     class_name = class_names[int(class_id)-1]
                    #     color = COLORS[int(class_id)]
                    #     # get the bounding box coordinates
                    #     box_x = det[3] * image_width
                    #     box_y = det[4] * image_height
                    #     # get the bounding box width and height
                    #     box_width = det[5] * image_width
                    #     box_height = det[6] * image_height
                    #     # draw a rectangle around each detected object
                    #     cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
                    #     # put the class name text on the detected object
                    #     cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    # if det is None:
                    #     print('no detection...hovering')
                    #     hover(cf)
                    # else:
                        # print('detection...tracking')
                        # _, _, _, box_x, box_y, box_width, box_height = det
                        # box_x, box_y = detection_center(det)
                        # exit_loop, x_cur, y_cur = move_to_book(cf, box_x, box_y, box_width, box_height, x_cur, y_cur)