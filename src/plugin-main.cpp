/*
Plugin Name
Copyright (C) <Year> <Developer> <Email Address>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/

#include <obs-module.h>
#include <plugin-support.h>

#include <opencv2/core/core.hpp>													  
#include <opencv2/highgui/highgui.hpp>												
#include <opencv2/imgproc/imgproc.hpp>												
#include <opencv2/objdetect/objdetect.hpp>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

cv::CascadeClassifier classifier;
cv::Mat hat;

char *classifierpath = NULL;
char *hatpath = NULL;

static obs_properties_t *hat_source_properties(void *data);
static struct obs_source_frame *hat_filter_video(void *data, struct obs_source_frame *frame);

void* hat_create(obs_data_t *settings, obs_source_t *context) {
	obs_data_set_default_string(settings, "haar", "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt_tree.xml");

	return (void*)context;
}

void hat_update(void *data, obs_data_t *settings) {
	(void)data;

	classifierpath = (char*)obs_data_get_string(settings, "haar");
	hatpath = (char*)obs_data_get_string(settings, "hat");
}

struct obs_source_info hat_frame_filter_info = {
	.id	   	= "hat_frame_filter",
	.type	 	= OBS_SOURCE_TYPE_FILTER,
	.output_flags	= OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC,
	.get_name	= [](void*) { return "Hat Frame Filter"; },
	.create	   	= hat_create,
	.destroy	= [](void *data) { (void)data; },
	.get_properties	= hat_source_properties,
	.update         = hat_update,
	.filter_video 	= hat_filter_video,
};

void putHatOn(cv::Mat &frame)
{
	int x, y;
	cv::Mat grayscale;
	cv::Mat green(frame.rows*2, frame.cols*3, CV_8UC3, cv::Scalar(0,0,0));
	cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(grayscale, grayscale); // enhance image contrast
	std::vector<cv::Rect> faces;
	static std::vector<cv::Rect> lastfaces;
	static int lastfacescount = 0;
	classifier.detectMultiScale(grayscale, faces, 1.01, 1, 0, cv::Size(90, 90), cv::Size());
	if (faces.size() == 0) {
		lastfacescount++;
		if (lastfacescount > 40)
			lastfaces = faces;
		if (lastfaces.size() == 0)
			return;
		faces = lastfaces;
	} else
		lastfacescount = 0;

	double width = faces[0].br().x - faces[0].tl().x;
	width *= 2;
	double ratio = width / hat.cols;
	double height = ratio * hat.rows;
	cv::Mat resizedHat;
	cv::resize(hat, resizedHat, cv::Size(width, height), ratio, ratio);
	cv::Rect hatRect(faces[0].tl().x + frame.cols - width/4, faces[0].tl().y - 3*height/4 + frame.rows, resizedHat.cols, resizedHat.rows);
	resizedHat.copyTo(green(hatRect));

	for (y = 0; y < frame.rows; y++) for (x = 0; x < frame.cols; x++) {
		cv::Vec3b color = green.at<cv::Vec3b>(cv::Point(x + frame.cols,y + frame.rows));
		if (color[0] != 0 || color[1] != 0 || color[2] != 0)
			frame.at<cv::Vec3b>(cv::Point(x,y)) = color;
	}

	lastfaces = faces;
}

static struct obs_source_frame *hat_filter_video(void *data, struct obs_source_frame *frame)
{
	// Access plane 0 (e.g., the Y channel in NV12 or the entire buffer in RGBA)
	uint8_t *pixel_data = frame->data[0];
	uint32_t stride = frame->linesize[0];
	static bool loaded;

	(void)data;

	if (hatpath != NULL) {
		hat = cv::imread(hatpath);

		hatpath = NULL;

		if (!hat.data) {
			return frame;
		}
	}

	if (classifierpath != NULL) {
		loaded = classifier.load(classifierpath);

		obs_log(LOG_INFO, "classifier.load: %s - %s", classifierpath, loaded? "success": "failure");

		classifierpath = NULL;

		if (loaded == false) {
			return frame;
		}
	}

	cv::Mat cvmat(frame->height * 2, frame->width * 3, CV_8UC3, cv::Scalar(0,0,0));

	// Example frame manipulation: Iterate through rows and columns
	for (uint32_t y = 0; y < frame->height; y++) {
		for (uint32_t x = 0; x < frame->width; x++) {
			size_t pixel_index = (y * stride) + (x * 3);
			cv::Vec3b color = cv::Vec3b(pixel_data[pixel_index], pixel_data[pixel_index+1], pixel_data[pixel_index+2]);
			cvmat.at<cv::Vec3b>(cv::Point(x,y)) = color;
		}
	}

	if (hat.data != NULL && loaded == true)
		putHatOn(cvmat);

	for (uint32_t y = 0; y < frame->height; y++) {
		for (uint32_t x = 0; x < frame->width; x++) {
			cv::Vec3b color = cvmat.at<cv::Vec3b>(cv::Point(x,y));
			size_t pixel_index = (y * stride) + (x * 3);
			pixel_data[pixel_index  ] = color[0];
			pixel_data[pixel_index+1] = color[1];
			pixel_data[pixel_index+2] = color[2];
		}
	}

	// Return the altered frame back to the OBS compositing engine
	return frame; 
}

static obs_properties_t *hat_source_properties(void *data)
{
	obs_properties_t *ppts = obs_properties_create();

	(void)data;

	obs_properties_add_text(ppts, "haar", "Haar Cascade File Location", OBS_TEXT_DEFAULT);
	obs_properties_add_text(ppts, "hat", "Hat Image File Location", OBS_TEXT_DEFAULT);

	return ppts;
}

bool obs_module_load(void)
{
	obs_register_source(&hat_frame_filter_info);

	obs_log(LOG_INFO, "plugin loaded successfully (version %s)", PLUGIN_VERSION);

	return true;
}

void obs_module_unload(void)
{
	obs_log(LOG_INFO, "plugin unloaded");
}
