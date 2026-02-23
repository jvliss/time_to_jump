/************* 
 * Main Test *
 *************/

import { core, data, sound, util, visual } from './lib/psychojs-2022.1.4.js';
const { PsychoJS } = core;
const { TrialHandler, MultiStairHandler } = data;
const { Scheduler } = util;
//some handy aliases as in the psychopy scripts;
const { abs, sin, cos, PI: pi, sqrt } = Math;
const { round } = util;


// store info about the experiment session:
let expName = 'main';  // from the Builder filename that created this script
let expInfo = {'pp': '1'};

// Start code blocks for 'Before Experiment'
var image_size = [0.7, 0.7];
var condition = ["Hell", "Dunkel"];
var adjective = ["heller", "dunkler"];
var key_assigned = ["A", "L"];
var hell = "a";
var dunkel = "l";
var prop = "51.5";

var deadline = 99999;
var deadline_test = 99999;

var groups = ["acc", "acc", "acc"]; //["control", "acc", "gutfeeling2"]; //
var group = "control"; //"acc"; //"control //gutfeeling2

var imgs = []

var safari = false;
var opera = false;


function isFullscreen(){
    var isInFullScreen = (document.fullscreenElement && document.fullscreenElement !== null) ||
        (document.webkitFullscreenElement && document.webkitFullscreenElement !== null) ||
        (document.mozFullScreenElement && document.mozFullScreenElement !== null) ||
        (document.msFullscreenElement && document.msFullscreenElement !== null);
        return isInFullScreen
}


function isBrowserUpToDate() {
  var userAgent = navigator.userAgent;

  // Check if the browser is Chrome
  if (/Chrome\/(\d+)/.test(userAgent)) {
    // Get the version number of Chrome
    var version = parseInt(RegExp.$1);
    // Check if the version of Chrome is at least 109
    if (version >= 109) {
      return true;
    }
  }
  // Check if the browser is Firefox
  else if (/Firefox\/(\d+)/.test(userAgent)) {
    // Get the version number of Firefox
    var version = parseInt(RegExp.$1);
    // Check if the version of Firefox is at least 102
    if (version >= 102) {
      return true;
    }
  }
  // Check if the browser is Safari
  else if (/^Mozilla\/5.0 \(.+?\) AppleWebKit\/(\d+)\./.test(userAgent)) {
    safari = true;
    // Get the version number of Safari
    var version = parseInt(RegExp.$1);
    // Check if the version of Safari is at least 16
    if (version >= 16) {
      return true;
    }
  }
  // Check if the browser is Edge
  else if (/Edg\/(\d+)/.test(userAgent)) {
    // Get the version number of Edge
    var version = parseInt(RegExp.$1);
    // Check if the version of Edge is at least 110
    if (version >= 110) {
      return true;
    }
  }

  // The browser is not up to date
  return false;
}
var browser_check 
var userAgent
var text_out = ""
var text_out_time = 0
var text_end = 'Vielen Dank für Ihre Zeit.'
var text_end_time = 0
var feedback_length = 1.2;
var ISI_length = 0.3;

var block_i = -1;
var block_images = [];
var block_cutoffs = [];
var trial_i = 0;
var textsize = 0.025;
var upper_text = 0.415;
var x_scale = 0.05;
var y_scale = 0.05;

var color_false = [0.7255, -0.8431, -0.5294];
var color_correct = [-1.0000, 0.0039, -1.0000];

var group_instruction1 = "Nun gilt zudem:" + "\n" + "Bitte versuchen Sie, die Aufgabe gut zu bearbeiten: Es geht darum, so schnell wie möglich zu antworten und gleichzeitig möglichst wenige Fehler zu machen.";
var group_instruction2 = "Welche Fläche ist insgesamt größer?";
var group_instruction3 = "wenn Sie einen Fehler gemacht haben.";

var image_path = ''

var version = 1
var feedback_text = "Falsch!"
var feedback_color = color_false
var acc_responses = 0
var delayed_responses = 0
var in_time_responses = 0

var condition_met_acc = 0
var condition_met_fullscreen = 0
var condition_met = 0

var fully_screenys = 0
// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([0,0,0]),
  units: 'height',
  waitBlanking: true
});
// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); }, flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(setupRoutineBegin());
flowScheduler.add(setupRoutineEachFrame());
flowScheduler.add(setupRoutineEnd());
flowScheduler.add(browser_outRoutineBegin());
flowScheduler.add(browser_outRoutineEachFrame());
flowScheduler.add(browser_outRoutineEnd());
flowScheduler.add(scalingRoutineBegin());
flowScheduler.add(scalingRoutineEachFrame());
flowScheduler.add(scalingRoutineEnd());
flowScheduler.add(premature_endRoutineBegin());
flowScheduler.add(premature_endRoutineEachFrame());
flowScheduler.add(premature_endRoutineEnd());
flowScheduler.add(instruction1RoutineBegin());
flowScheduler.add(instruction1RoutineEachFrame());
flowScheduler.add(instruction1RoutineEnd());
flowScheduler.add(instruction2RoutineBegin());
flowScheduler.add(instruction2RoutineEachFrame());
flowScheduler.add(instruction2RoutineEnd());
flowScheduler.add(prepRoutineBegin());
flowScheduler.add(prepRoutineEachFrame());
flowScheduler.add(prepRoutineEnd());
flowScheduler.add(countdownRoutineBegin());
flowScheduler.add(countdownRoutineEachFrame());
flowScheduler.add(countdownRoutineEnd());
const practice_trialsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(practice_trialsLoopBegin(practice_trialsLoopScheduler));
flowScheduler.add(practice_trialsLoopScheduler);
flowScheduler.add(practice_trialsLoopEnd);
flowScheduler.add(premature_endRoutineBegin());
flowScheduler.add(premature_endRoutineEachFrame());
flowScheduler.add(premature_endRoutineEnd());
flowScheduler.add(instruction3RoutineBegin());
flowScheduler.add(instruction3RoutineEachFrame());
flowScheduler.add(instruction3RoutineEnd());
flowScheduler.add(instruction4RoutineBegin());
flowScheduler.add(instruction4RoutineEachFrame());
flowScheduler.add(instruction4RoutineEnd());
const test_blocksLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(test_blocksLoopBegin(test_blocksLoopScheduler));
flowScheduler.add(test_blocksLoopScheduler);
flowScheduler.add(test_blocksLoopEnd);
flowScheduler.add(quitPsychoJS, '', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, '', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    {'name': 'imgs/blob_107_r1.png', 'path': 'imgs/blob_107_r1.png'},
    {'name': 'imgs/blob_242_r2.png', 'path': 'imgs/blob_242_r2.png'},
    {'name': 'imgs/blob_3_r3.png', 'path': 'imgs/blob_3_r3.png'},
    {'name': 'imgs/blob_37_r2.png', 'path': 'imgs/blob_37_r2.png'},
    {'name': 'imgs/blob_12_r2.png', 'path': 'imgs/blob_12_r2.png'},
    {'name': 'imgs/blob_13.png', 'path': 'imgs/blob_13.png'},
    {'name': 'rsc/bank-1300155_640.png', 'path': 'rsc/bank-1300155_640.png'},
    {'name': 'imgs/blob_345_r2.png', 'path': 'imgs/blob_345_r2.png'},
    {'name': 'imgs/blob_0_r3.png', 'path': 'imgs/blob_0_r3.png'},
    {'name': 'imgs/blob_9.png', 'path': 'imgs/blob_9.png'},
    {'name': 'imgs/blob_308_r3.png', 'path': 'imgs/blob_308_r3.png'},
    {'name': 'imgs/blob_404.png', 'path': 'imgs/blob_404.png'},
    {'name': 'imgs/blob_202.png', 'path': 'imgs/blob_202.png'},
    {'name': 'imgs/blob_257.png', 'path': 'imgs/blob_257.png'},
    {'name': 'imgs/blob_66_r1.png', 'path': 'imgs/blob_66_r1.png'},
    {'name': 'imgs/blob_347.png', 'path': 'imgs/blob_347.png'},
    {'name': 'imgs/blob_395.png', 'path': 'imgs/blob_395.png'},
    {'name': 'imgs/blob_313_r2.png', 'path': 'imgs/blob_313_r2.png'},
    {'name': 'imgs/blob_269.png', 'path': 'imgs/blob_269.png'},
    {'name': 'imgs/blob_37_r3.png', 'path': 'imgs/blob_37_r3.png'},
    {'name': 'imgs/blob_7_r1.png', 'path': 'imgs/blob_7_r1.png'},
    {'name': 'imgs/blob_4.png', 'path': 'imgs/blob_4.png'},
    {'name': 'imgs/blob_57.png', 'path': 'imgs/blob_57.png'},
    {'name': 'imgs/blob_204.png', 'path': 'imgs/blob_204.png'},
    {'name': 'imgs/blob_103.png', 'path': 'imgs/blob_103.png'},
    {'name': 'imgs/blob_57_r3.png', 'path': 'imgs/blob_57_r3.png'},
    {'name': 'imgs/blob_363_r2.png', 'path': 'imgs/blob_363_r2.png'},
    {'name': 'imgs/blob_377.png', 'path': 'imgs/blob_377.png'},
    {'name': 'imgs/blob_189_r2.png', 'path': 'imgs/blob_189_r2.png'},
    {'name': 'imgs/blob_239_r1.png', 'path': 'imgs/blob_239_r1.png'},
    {'name': 'imgs/blob_287_r1.png', 'path': 'imgs/blob_287_r1.png'},
    {'name': 'imgs/blob_278.png', 'path': 'imgs/blob_278.png'},
    {'name': 'imgs/blob_151_r1.png', 'path': 'imgs/blob_151_r1.png'},
    {'name': 'imgs/blob_72_r3.png', 'path': 'imgs/blob_72_r3.png'},
    {'name': 'imgs/blob_202_r1.png', 'path': 'imgs/blob_202_r1.png'},
    {'name': 'imgs/blob_294.png', 'path': 'imgs/blob_294.png'},
    {'name': 'imgs/blob_88_r3.png', 'path': 'imgs/blob_88_r3.png'},
    {'name': 'imgs/blob_289.png', 'path': 'imgs/blob_289.png'},
    {'name': 'imgs/blob_322_r1.png', 'path': 'imgs/blob_322_r1.png'},
    {'name': 'imgs/blob_0_r2.png', 'path': 'imgs/blob_0_r2.png'},
    {'name': 'imgs/blob_91_r1.png', 'path': 'imgs/blob_91_r1.png'},
    {'name': 'imgs/blob_267.png', 'path': 'imgs/blob_267.png'},
    {'name': 'imgs/blob_66.png', 'path': 'imgs/blob_66.png'},
    {'name': 'imgs/blob_80_r3.png', 'path': 'imgs/blob_80_r3.png'},
    {'name': 'imgs/blob_89_r1.png', 'path': 'imgs/blob_89_r1.png'},
    {'name': 'imgs/blob_355_r3.png', 'path': 'imgs/blob_355_r3.png'},
    {'name': 'imgs/blob_89.png', 'path': 'imgs/blob_89.png'},
    {'name': 'imgs/blob_324_r1.png', 'path': 'imgs/blob_324_r1.png'},
    {'name': 'imgs/blob_69_r2.png', 'path': 'imgs/blob_69_r2.png'},
    {'name': 'imgs/blob_187.png', 'path': 'imgs/blob_187.png'},
    {'name': 'imgs/blob_320.png', 'path': 'imgs/blob_320.png'},
    {'name': 'imgs/blob_91.png', 'path': 'imgs/blob_91.png'},
    {'name': 'imgs/blob_135.png', 'path': 'imgs/blob_135.png'},
    {'name': 'imgs/blob_308.png', 'path': 'imgs/blob_308.png'},
    {'name': 'imgs/blob_294_r3.png', 'path': 'imgs/blob_294_r3.png'},
    {'name': 'imgs/blob_267_r1.png', 'path': 'imgs/blob_267_r1.png'},
    {'name': 'imgs/blob_153_r1.png', 'path': 'imgs/blob_153_r1.png'},
    {'name': 'imgs/blob_56.png', 'path': 'imgs/blob_56.png'},
    {'name': 'imgs/blob_275_r1.png', 'path': 'imgs/blob_275_r1.png'},
    {'name': 'imgs/blob_312.png', 'path': 'imgs/blob_312.png'},
    {'name': 'imgs/blob_89_r3.png', 'path': 'imgs/blob_89_r3.png'},
    {'name': 'imgs/blob_0_r1.png', 'path': 'imgs/blob_0_r1.png'},
    {'name': 'imgs/blob_308_r2.png', 'path': 'imgs/blob_308_r2.png'},
    {'name': 'imgs/blob_193_r3.png', 'path': 'imgs/blob_193_r3.png'},
    {'name': 'imgs/blob_125_r1.png', 'path': 'imgs/blob_125_r1.png'},
    {'name': 'imgs/blob_310_r2.png', 'path': 'imgs/blob_310_r2.png'},
    {'name': 'imgs/blob_1_r1.png', 'path': 'imgs/blob_1_r1.png'},
    {'name': 'imgs/blob_77_r1.png', 'path': 'imgs/blob_77_r1.png'},
    {'name': 'imgs/blob_345.png', 'path': 'imgs/blob_345.png'},
    {'name': 'imgs/blob_30_r1.png', 'path': 'imgs/blob_30_r1.png'},
    {'name': 'imgs/blob_287.png', 'path': 'imgs/blob_287.png'},
    {'name': 'imgs/blob_80_r1.png', 'path': 'imgs/blob_80_r1.png'},
    {'name': 'imgs/blob_394.png', 'path': 'imgs/blob_394.png'},
    {'name': 'imgs/blob_189.png', 'path': 'imgs/blob_189.png'},
    {'name': 'imgs/blob_125_r2.png', 'path': 'imgs/blob_125_r2.png'},
    {'name': 'imgs/blob_263_r1.png', 'path': 'imgs/blob_263_r1.png'},
    {'name': 'imgs/blob_107_r3.png', 'path': 'imgs/blob_107_r3.png'},
    {'name': 'imgs/blob_100.png', 'path': 'imgs/blob_100.png'},
    {'name': 'imgs/blob_56_r3.png', 'path': 'imgs/blob_56_r3.png'},
    {'name': 'imgs/blob_30.png', 'path': 'imgs/blob_30.png'},
    {'name': 'imgs/blob_137.png', 'path': 'imgs/blob_137.png'},
    {'name': 'imgs/blob_296_r1.png', 'path': 'imgs/blob_296_r1.png'},
    {'name': 'imgs/blob_239.png', 'path': 'imgs/blob_239.png'},
    {'name': 'imgs/blob_371_r3.png', 'path': 'imgs/blob_371_r3.png'},
    {'name': 'imgs/blob_392.png', 'path': 'imgs/blob_392.png'},
    {'name': 'imgs/blob_143_r3.png', 'path': 'imgs/blob_143_r3.png'},
    {'name': 'imgs/blob_66_r2.png', 'path': 'imgs/blob_66_r2.png'},
    {'name': 'imgs/blob_362_r1.png', 'path': 'imgs/blob_362_r1.png'},
    {'name': 'imgs/blob_13_r3.png', 'path': 'imgs/blob_13_r3.png'},
    {'name': 'imgs/blob_124.png', 'path': 'imgs/blob_124.png'},
    {'name': 'imgs/blob_254_r1.png', 'path': 'imgs/blob_254_r1.png'},
    {'name': 'imgs/blob_72_r1.png', 'path': 'imgs/blob_72_r1.png'},
    {'name': 'imgs/blob_289_r2.png', 'path': 'imgs/blob_289_r2.png'},
    {'name': 'imgs/blob_35_r2.png', 'path': 'imgs/blob_35_r2.png'},
    {'name': 'imgs/blob_369.png', 'path': 'imgs/blob_369.png'},
    {'name': 'imgs/blob_37.png', 'path': 'imgs/blob_37.png'},
    {'name': 'imgs/blob_202_r3.png', 'path': 'imgs/blob_202_r3.png'},
    {'name': 'imgs/blob_279.png', 'path': 'imgs/blob_279.png'},
    {'name': 'imgs/blob_362.png', 'path': 'imgs/blob_362.png'},
    {'name': 'imgs/blob_400.png', 'path': 'imgs/blob_400.png'},
    {'name': 'imgs/blob_193_r1.png', 'path': 'imgs/blob_193_r1.png'},
    {'name': 'imgs/blob_397.png', 'path': 'imgs/blob_397.png'},
    {'name': 'imgs/blob_100_r2.png', 'path': 'imgs/blob_100_r2.png'},
    {'name': 'imgs/blob_211.png', 'path': 'imgs/blob_211.png'},
    {'name': 'imgs/blob_275_r3.png', 'path': 'imgs/blob_275_r3.png'},
    {'name': 'imgs/blob_359_r3.png', 'path': 'imgs/blob_359_r3.png'},
    {'name': 'imgs/blob_13_r1.png', 'path': 'imgs/blob_13_r1.png'},
    {'name': 'imgs/blob_263_r2.png', 'path': 'imgs/blob_263_r2.png'},
    {'name': 'imgs/blob_247_r3.png', 'path': 'imgs/blob_247_r3.png'},
    {'name': 'imgs/blob_87_r2.png', 'path': 'imgs/blob_87_r2.png'},
    {'name': 'imgs/blob_109_r2.png', 'path': 'imgs/blob_109_r2.png'},
    {'name': 'imgs/blob_165_r3.png', 'path': 'imgs/blob_165_r3.png'},
    {'name': 'imgs/blob_207.png', 'path': 'imgs/blob_207.png'},
    {'name': 'imgs/blob_366.png', 'path': 'imgs/blob_366.png'},
    {'name': 'imgs/blob_371_r2.png', 'path': 'imgs/blob_371_r2.png'},
    {'name': 'imgs/blob_13_r2.png', 'path': 'imgs/blob_13_r2.png'},
    {'name': 'imgs/blob_12_r3.png', 'path': 'imgs/blob_12_r3.png'},
    {'name': 'imgs/blob_100_r3.png', 'path': 'imgs/blob_100_r3.png'},
    {'name': 'imgs/blob_212.png', 'path': 'imgs/blob_212.png'},
    {'name': 'imgs/blob_67_r3.png', 'path': 'imgs/blob_67_r3.png'},
    {'name': 'imgs/blob_77_r2.png', 'path': 'imgs/blob_77_r2.png'},
    {'name': 'imgs/blob_339_r2.png', 'path': 'imgs/blob_339_r2.png'},
    {'name': 'imgs/blob_218.png', 'path': 'imgs/blob_218.png'},
    {'name': 'imgs/blob_125_r3.png', 'path': 'imgs/blob_125_r3.png'},
    {'name': 'imgs/blob_260_r2.png', 'path': 'imgs/blob_260_r2.png'},
    {'name': 'imgs/blob_1_r3.png', 'path': 'imgs/blob_1_r3.png'},
    {'name': 'imgs/blob_278_r3.png', 'path': 'imgs/blob_278_r3.png'},
    {'name': 'imgs/blob_355_r2.png', 'path': 'imgs/blob_355_r2.png'},
    {'name': 'imgs/blob_239_r2.png', 'path': 'imgs/blob_239_r2.png'},
    {'name': 'imgs/blob_302_r3.png', 'path': 'imgs/blob_302_r3.png'},
    {'name': 'imgs/blob_310.png', 'path': 'imgs/blob_310.png'},
    {'name': 'imgs/blob_280_r2.png', 'path': 'imgs/blob_280_r2.png'},
    {'name': 'imgs/blob_322_r3.png', 'path': 'imgs/blob_322_r3.png'},
    {'name': 'imgs/blob_37_r1.png', 'path': 'imgs/blob_37_r1.png'},
    {'name': 'imgs/blob_371_r1.png', 'path': 'imgs/blob_371_r1.png'},
    {'name': 'imgs/blob_3.png', 'path': 'imgs/blob_3.png'},
    {'name': 'imgs/blob_249.png', 'path': 'imgs/blob_249.png'},
    {'name': 'imgs/blob_15_r3.png', 'path': 'imgs/blob_15_r3.png'},
    {'name': 'imgs/blob_66_r3.png', 'path': 'imgs/blob_66_r3.png'},
    {'name': 'imgs/blob_279_r1.png', 'path': 'imgs/blob_279_r1.png'},
    {'name': 'imgs/blob_180.png', 'path': 'imgs/blob_180.png'},
    {'name': 'imgs/blob_58.png', 'path': 'imgs/blob_58.png'},
    {'name': 'imgs/blob_339_r1.png', 'path': 'imgs/blob_339_r1.png'},
    {'name': 'imgs/blob_280_r3.png', 'path': 'imgs/blob_280_r3.png'},
    {'name': 'imgs/blob_296_r2.png', 'path': 'imgs/blob_296_r2.png'},
    {'name': 'imgs/blob_35.png', 'path': 'imgs/blob_35.png'},
    {'name': 'imgs/blob_257_r3.png', 'path': 'imgs/blob_257_r3.png'},
    {'name': 'imgs/blob_137_r3.png', 'path': 'imgs/blob_137_r3.png'},
    {'name': 'imgs/blob_295_r3.png', 'path': 'imgs/blob_295_r3.png'},
    {'name': 'imgs/blob_193_r2.png', 'path': 'imgs/blob_193_r2.png'},
    {'name': 'imgs/blob_188_r3.png', 'path': 'imgs/blob_188_r3.png'},
    {'name': 'imgs/blob_188_r1.png', 'path': 'imgs/blob_188_r1.png'},
    {'name': 'imgs/blob_56_r1.png', 'path': 'imgs/blob_56_r1.png'},
    {'name': 'imgs/blob_378_r2.png', 'path': 'imgs/blob_378_r2.png'},
    {'name': 'imgs/blob_311_r1.png', 'path': 'imgs/blob_311_r1.png'},
    {'name': 'imgs/blob_369_r3.png', 'path': 'imgs/blob_369_r3.png'},
    {'name': 'imgs/blob_324_r3.png', 'path': 'imgs/blob_324_r3.png'},
    {'name': 'imgs/blob_109.png', 'path': 'imgs/blob_109.png'},
    {'name': 'imgs/blob_180_r2.png', 'path': 'imgs/blob_180_r2.png'},
    {'name': 'imgs/blob_311_r2.png', 'path': 'imgs/blob_311_r2.png'},
    {'name': 'imgs/blob_87.png', 'path': 'imgs/blob_87.png'},
    {'name': 'imgs/blob_396.png', 'path': 'imgs/blob_396.png'},
    {'name': 'imgs/blob_49_r1.png', 'path': 'imgs/blob_49_r1.png'},
    {'name': 'imgs/blob_3_r1.png', 'path': 'imgs/blob_3_r1.png'},
    {'name': 'imgs/blob_381.png', 'path': 'imgs/blob_381.png'},
    {'name': 'imgs/blob_129_r2.png', 'path': 'imgs/blob_129_r2.png'},
    {'name': 'imgs/blob_271_r3.png', 'path': 'imgs/blob_271_r3.png'},
    {'name': 'imgs/blob_207_r2.png', 'path': 'imgs/blob_207_r2.png'},
    {'name': 'imgs/blob_12.png', 'path': 'imgs/blob_12.png'},
    {'name': 'imgs/blob_355_r1.png', 'path': 'imgs/blob_355_r1.png'},
    {'name': 'imgs/blob_131.png', 'path': 'imgs/blob_131.png'},
    {'name': 'imgs/blob_7_r3.png', 'path': 'imgs/blob_7_r3.png'},
    {'name': 'imgs/blob_202_r2.png', 'path': 'imgs/blob_202_r2.png'},
    {'name': 'imgs/blob_359_r1.png', 'path': 'imgs/blob_359_r1.png'},
    {'name': 'imgs/blob_151.png', 'path': 'imgs/blob_151.png'},
    {'name': 'imgs/blob_72_r2.png', 'path': 'imgs/blob_72_r2.png'},
    {'name': 'imgs/blob_278_r2.png', 'path': 'imgs/blob_278_r2.png'},
    {'name': 'imgs/blob_18.png', 'path': 'imgs/blob_18.png'},
    {'name': 'imgs/blob_295.png', 'path': 'imgs/blob_295.png'},
    {'name': 'imgs/blob_98.png', 'path': 'imgs/blob_98.png'},
    {'name': 'imgs/blob_296.png', 'path': 'imgs/blob_296.png'},
    {'name': 'imgs/blob_165_r1.png', 'path': 'imgs/blob_165_r1.png'},
    {'name': 'imgs/blob_312_r1.png', 'path': 'imgs/blob_312_r1.png'},
    {'name': 'imgs/blob_135_r2.png', 'path': 'imgs/blob_135_r2.png'},
    {'name': 'imgs/blob_371.png', 'path': 'imgs/blob_371.png'},
    {'name': 'imgs/blob_260_r3.png', 'path': 'imgs/blob_260_r3.png'},
    {'name': 'imgs/blob_393.png', 'path': 'imgs/blob_393.png'},
    {'name': 'imgs/blob_295_r2.png', 'path': 'imgs/blob_295_r2.png'},
    {'name': 'imgs/blob_322.png', 'path': 'imgs/blob_322.png'},
    {'name': 'imgs/blob_388.png', 'path': 'imgs/blob_388.png'},
    {'name': 'imgs/blob_204_r1.png', 'path': 'imgs/blob_204_r1.png'},
    {'name': 'imgs/blob_0.png', 'path': 'imgs/blob_0.png'},
    {'name': 'imgs/blob_247_r1.png', 'path': 'imgs/blob_247_r1.png'},
    {'name': 'imgs/blob_271_r2.png', 'path': 'imgs/blob_271_r2.png'},
    {'name': 'imgs/blob_1.png', 'path': 'imgs/blob_1.png'},
    {'name': 'imgs/blob_302.png', 'path': 'imgs/blob_302.png'},
    {'name': 'imgs/blob_289_r3.png', 'path': 'imgs/blob_289_r3.png'},
    {'name': 'imgs/blob_151_r2.png', 'path': 'imgs/blob_151_r2.png'},
    {'name': 'imgs/blob_119_r1.png', 'path': 'imgs/blob_119_r1.png'},
    {'name': 'imgs/blob_207_r1.png', 'path': 'imgs/blob_207_r1.png'},
    {'name': 'imgs/blob_362_r3.png', 'path': 'imgs/blob_362_r3.png'},
    {'name': 'imgs/blob_77_r3.png', 'path': 'imgs/blob_77_r3.png'},
    {'name': 'imgs/blob_166.png', 'path': 'imgs/blob_166.png'},
    {'name': 'imgs/blob_322_r2.png', 'path': 'imgs/blob_322_r2.png'},
    {'name': 'imgs/blob_30_r3.png', 'path': 'imgs/blob_30_r3.png'},
    {'name': 'imgs/blob_250_r3.png', 'path': 'imgs/blob_250_r3.png'},
    {'name': 'imgs/blob_313_r3.png', 'path': 'imgs/blob_313_r3.png'},
    {'name': 'imgs/blob_355.png', 'path': 'imgs/blob_355.png'},
    {'name': 'imgs/blob_72.png', 'path': 'imgs/blob_72.png'},
    {'name': 'imgs/blob_401.png', 'path': 'imgs/blob_401.png'},
    {'name': 'imgs/blob_207_r3.png', 'path': 'imgs/blob_207_r3.png'},
    {'name': 'imgs/blob_378_r3.png', 'path': 'imgs/blob_378_r3.png'},
    {'name': 'imgs/blob_339_r3.png', 'path': 'imgs/blob_339_r3.png'},
    {'name': 'imgs/blob_187_r3.png', 'path': 'imgs/blob_187_r3.png'},
    {'name': 'imgs/blob_378.png', 'path': 'imgs/blob_378.png'},
    {'name': 'imgs/blob_324.png', 'path': 'imgs/blob_324.png'},
    {'name': 'imgs/blob_254_r3.png', 'path': 'imgs/blob_254_r3.png'},
    {'name': 'imgs/blob_119.png', 'path': 'imgs/blob_119.png'},
    {'name': 'imgs/blob_215.png', 'path': 'imgs/blob_215.png'},
    {'name': 'imgs/blob_171_r1.png', 'path': 'imgs/blob_171_r1.png'},
    {'name': 'imgs/blob_215_r1.png', 'path': 'imgs/blob_215_r1.png'},
    {'name': 'imgs/blob_359.png', 'path': 'imgs/blob_359.png'},
    {'name': 'imgs/blob_215_r2.png', 'path': 'imgs/blob_215_r2.png'},
    {'name': 'imgs/blob_294_r1.png', 'path': 'imgs/blob_294_r1.png'},
    {'name': 'imgs/blob_187_r1.png', 'path': 'imgs/blob_187_r1.png'},
    {'name': 'imgs/blob_257_r1.png', 'path': 'imgs/blob_257_r1.png'},
    {'name': 'imgs/blob_171_r3.png', 'path': 'imgs/blob_171_r3.png'},
    {'name': 'imgs/blob_129.png', 'path': 'imgs/blob_129.png'},
    {'name': 'imgs/blob_398.png', 'path': 'imgs/blob_398.png'},
    {'name': 'imgs/blob_279_r2.png', 'path': 'imgs/blob_279_r2.png'},
    {'name': 'imgs/blob_80_r2.png', 'path': 'imgs/blob_80_r2.png'},
    {'name': 'imgs/blob_267_r3.png', 'path': 'imgs/blob_267_r3.png'},
    {'name': 'imgs/blob_242_r1.png', 'path': 'imgs/blob_242_r1.png'},
    {'name': 'imgs/blob_389.png', 'path': 'imgs/blob_389.png'},
    {'name': 'imgs/blob_7_r2.png', 'path': 'imgs/blob_7_r2.png'},
    {'name': 'imgs/blob_269_r1.png', 'path': 'imgs/blob_269_r1.png'},
    {'name': 'imgs/blob_151_r3.png', 'path': 'imgs/blob_151_r3.png'},
    {'name': 'imgs/blob_279_r3.png', 'path': 'imgs/blob_279_r3.png'},
    {'name': 'imgs/blob_88_r2.png', 'path': 'imgs/blob_88_r2.png'},
    {'name': 'imgs/blob_171_r2.png', 'path': 'imgs/blob_171_r2.png'},
    {'name': 'imgs/blob_185_r2.png', 'path': 'imgs/blob_185_r2.png'},
    {'name': 'imgs/blob_363.png', 'path': 'imgs/blob_363.png'},
    {'name': 'imgs/blob_129_r3.png', 'path': 'imgs/blob_129_r3.png'},
    {'name': 'imgs/blob_119_r2.png', 'path': 'imgs/blob_119_r2.png'},
    {'name': 'imgs/blob_152_r2.png', 'path': 'imgs/blob_152_r2.png'},
    {'name': 'imgs/blob_403.png', 'path': 'imgs/blob_403.png'},
    {'name': 'imgs/blob_15_r2.png', 'path': 'imgs/blob_15_r2.png'},
    {'name': 'imgs/blob_267_r2.png', 'path': 'imgs/blob_267_r2.png'},
    {'name': 'imgs/blob_295_r1.png', 'path': 'imgs/blob_295_r1.png'},
    {'name': 'imgs/blob_313_r1.png', 'path': 'imgs/blob_313_r1.png'},
    {'name': 'imgs/blob_263_r3.png', 'path': 'imgs/blob_263_r3.png'},
    {'name': 'imgs/blob_275_r2.png', 'path': 'imgs/blob_275_r2.png'},
    {'name': 'imgs/blob_189_r1.png', 'path': 'imgs/blob_189_r1.png'},
    {'name': 'imgs/blob_91_r3.png', 'path': 'imgs/blob_91_r3.png'},
    {'name': 'imgs/blob_153.png', 'path': 'imgs/blob_153.png'},
    {'name': 'imgs/blob_363_r1.png', 'path': 'imgs/blob_363_r1.png'},
    {'name': 'imgs/blob_391.png', 'path': 'imgs/blob_391.png'},
    {'name': 'imgs/blob_359_r2.png', 'path': 'imgs/blob_359_r2.png'},
    {'name': 'imgs/blob_153_r2.png', 'path': 'imgs/blob_153_r2.png'},
    {'name': 'imgs/blob_15.png', 'path': 'imgs/blob_15.png'},
    {'name': 'imgs/blob_9_r3.png', 'path': 'imgs/blob_9_r3.png'},
    {'name': 'imgs/blob_278_r1.png', 'path': 'imgs/blob_278_r1.png'},
    {'name': 'imgs/blob_362_r2.png', 'path': 'imgs/blob_362_r2.png'},
    {'name': 'imgs/blob_369_r2.png', 'path': 'imgs/blob_369_r2.png'},
    {'name': 'imgs/blob_254.png', 'path': 'imgs/blob_254.png'},
    {'name': 'imgs/blob_49.png', 'path': 'imgs/blob_49.png'},
    {'name': 'imgs/blob_56_r2.png', 'path': 'imgs/blob_56_r2.png'},
    {'name': 'imgs/blob_312_r3.png', 'path': 'imgs/blob_312_r3.png'},
    {'name': 'imgs/blob_91_r2.png', 'path': 'imgs/blob_91_r2.png'},
    {'name': 'imgs/blob_193.png', 'path': 'imgs/blob_193.png'},
    {'name': 'imgs/blob_242_r3.png', 'path': 'imgs/blob_242_r3.png'},
    {'name': 'imgs/blob_262_r2.png', 'path': 'imgs/blob_262_r2.png'},
    {'name': 'imgs/blob_12_r1.png', 'path': 'imgs/blob_12_r1.png'},
    {'name': 'imgs/blob_263.png', 'path': 'imgs/blob_263.png'},
    {'name': 'imgs/blob_302_r1.png', 'path': 'imgs/blob_302_r1.png'},
    {'name': 'imgs/blob_390.png', 'path': 'imgs/blob_390.png'},
    {'name': 'imgs/blob_406.png', 'path': 'imgs/blob_406.png'},
    {'name': 'imgs/blob_311_r3.png', 'path': 'imgs/blob_311_r3.png'},
    {'name': 'imgs/blob_4_r2.png', 'path': 'imgs/blob_4_r2.png'},
    {'name': 'imgs/blob_345_r3.png', 'path': 'imgs/blob_345_r3.png'},
    {'name': 'imgs/blob_269_r3.png', 'path': 'imgs/blob_269_r3.png'},
    {'name': 'imgs/blob_339.png', 'path': 'imgs/blob_339.png'},
    {'name': 'imgs/blob_180_r3.png', 'path': 'imgs/blob_180_r3.png'},
    {'name': 'imgs/blob_185_r1.png', 'path': 'imgs/blob_185_r1.png'},
    {'name': 'imgs/blob_30_r2.png', 'path': 'imgs/blob_30_r2.png'},
    {'name': 'imgs/blob_15_r1.png', 'path': 'imgs/blob_15_r1.png'},
    {'name': 'imgs/blob_1_r2.png', 'path': 'imgs/blob_1_r2.png'},
    {'name': 'imgs/blob_7.png', 'path': 'imgs/blob_7.png'},
    {'name': 'imgs/blob_88.png', 'path': 'imgs/blob_88.png'},
    {'name': 'imgs/blob_74.png', 'path': 'imgs/blob_74.png'},
    {'name': 'imgs/blob_250.png', 'path': 'imgs/blob_250.png'},
    {'name': 'imgs/blob_152_r1.png', 'path': 'imgs/blob_152_r1.png'},
    {'name': 'imgs/blob_87_r1.png', 'path': 'imgs/blob_87_r1.png'},
    {'name': 'imgs/blob_188_r2.png', 'path': 'imgs/blob_188_r2.png'},
    {'name': 'imgs/blob_254_r2.png', 'path': 'imgs/blob_254_r2.png'},
    {'name': 'imgs/blob_143_r1.png', 'path': 'imgs/blob_143_r1.png'},
    {'name': 'imgs/blob_171.png', 'path': 'imgs/blob_171.png'},
    {'name': 'imgs/blob_312_r2.png', 'path': 'imgs/blob_312_r2.png'},
    {'name': 'imgs/blob_262_r1.png', 'path': 'imgs/blob_262_r1.png'},
    {'name': 'imgs/blob_271_r1.png', 'path': 'imgs/blob_271_r1.png'},
    {'name': 'imgs/blob_87_r3.png', 'path': 'imgs/blob_87_r3.png'},
    {'name': 'imgs/blob_67_r2.png', 'path': 'imgs/blob_67_r2.png'},
    {'name': 'imgs/blob_35_r1.png', 'path': 'imgs/blob_35_r1.png'},
    {'name': 'imgs/blob_280_r1.png', 'path': 'imgs/blob_280_r1.png'},
    {'name': 'imgs/blob_187_r2.png', 'path': 'imgs/blob_187_r2.png'},
    {'name': 'imgs/blob_109_r3.png', 'path': 'imgs/blob_109_r3.png'},
    {'name': 'imgs/blob_369_r1.png', 'path': 'imgs/blob_369_r1.png'},
    {'name': 'imgs/blob_239_r3.png', 'path': 'imgs/blob_239_r3.png'},
    {'name': 'imgs/blob_363_r3.png', 'path': 'imgs/blob_363_r3.png'},
    {'name': 'imgs/blob_302_r2.png', 'path': 'imgs/blob_302_r2.png'},
    {'name': 'imgs/blob_107.png', 'path': 'imgs/blob_107.png'},
    {'name': 'imgs/blob_185_r3.png', 'path': 'imgs/blob_185_r3.png'},
    {'name': 'imgs/blob_165.png', 'path': 'imgs/blob_165.png'},
    {'name': 'imgs/blob_310_r1.png', 'path': 'imgs/blob_310_r1.png'},
    {'name': 'imgs/blob_69_r1.png', 'path': 'imgs/blob_69_r1.png'},
    {'name': 'imgs/blob_347_r3.png', 'path': 'imgs/blob_347_r3.png'},
    {'name': 'imgs/blob_9_r2.png', 'path': 'imgs/blob_9_r2.png'},
    {'name': 'imgs/blob_262.png', 'path': 'imgs/blob_262.png'},
    {'name': 'imgs/blob_407.png', 'path': 'imgs/blob_407.png'},
    {'name': 'imgs/blob_204_r2.png', 'path': 'imgs/blob_204_r2.png'},
    {'name': 'imgs/blob_287_r3.png', 'path': 'imgs/blob_287_r3.png'},
    {'name': 'imgs/blob_247.png', 'path': 'imgs/blob_247.png'},
    {'name': 'imgs/blob_67_r1.png', 'path': 'imgs/blob_67_r1.png'},
    {'name': 'imgs/blob_287_r2.png', 'path': 'imgs/blob_287_r2.png'},
    {'name': 'imgs/blob_321.png', 'path': 'imgs/blob_321.png'},
    {'name': 'imgs/blob_57_r2.png', 'path': 'imgs/blob_57_r2.png'},
    {'name': 'imgs/blob_399.png', 'path': 'imgs/blob_399.png'},
    {'name': 'imgs/blob_4_r3.png', 'path': 'imgs/blob_4_r3.png'},
    {'name': 'imgs/blob_69.png', 'path': 'imgs/blob_69.png'},
    {'name': 'imgs/blob_269_r2.png', 'path': 'imgs/blob_269_r2.png'},
    {'name': 'imgs/blob_262_r3.png', 'path': 'imgs/blob_262_r3.png'},
    {'name': 'imgs/blob_289_r1.png', 'path': 'imgs/blob_289_r1.png'},
    {'name': 'imgs/blob_89_r2.png', 'path': 'imgs/blob_89_r2.png'},
    {'name': 'imgs/blob_308_r1.png', 'path': 'imgs/blob_308_r1.png'},
    {'name': 'imgs/blob_137_r1.png', 'path': 'imgs/blob_137_r1.png'},
    {'name': 'imgs/blob_107_r2.png', 'path': 'imgs/blob_107_r2.png'},
    {'name': 'imgs/blob_3_r2.png', 'path': 'imgs/blob_3_r2.png'},
    {'name': 'imgs/blob_49_r3.png', 'path': 'imgs/blob_49_r3.png'},
    {'name': 'imgs/blob_114.png', 'path': 'imgs/blob_114.png'},
    {'name': 'imgs/blob_189_r3.png', 'path': 'imgs/blob_189_r3.png'},
    {'name': 'imgs/blob_188.png', 'path': 'imgs/blob_188.png'},
    {'name': 'imgs/blob_313.png', 'path': 'imgs/blob_313.png'},
    {'name': 'imgs/blob_135_r1.png', 'path': 'imgs/blob_135_r1.png'},
    {'name': 'imgs/blob_345_r1.png', 'path': 'imgs/blob_345_r1.png'},
    {'name': 'imgs/blob_125.png', 'path': 'imgs/blob_125.png'},
    {'name': 'imgs/blob_250_r1.png', 'path': 'imgs/blob_250_r1.png'},
    {'name': 'imgs/blob_143.png', 'path': 'imgs/blob_143.png'},
    {'name': 'imgs/blob_275.png', 'path': 'imgs/blob_275.png'},
    {'name': 'imgs/blob_378_r1.png', 'path': 'imgs/blob_378_r1.png'},
    {'name': 'imgs/blob_152_r3.png', 'path': 'imgs/blob_152_r3.png'},
    {'name': 'imgs/blob_296_r3.png', 'path': 'imgs/blob_296_r3.png'},
    {'name': 'imgs/blob_280.png', 'path': 'imgs/blob_280.png'},
    {'name': 'imgs/blob_67.png', 'path': 'imgs/blob_67.png'},
    {'name': 'imgs/blob_165_r2.png', 'path': 'imgs/blob_165_r2.png'},
    {'name': 'imgs/blob_137_r2.png', 'path': 'imgs/blob_137_r2.png'},
    {'name': 'imgs/blob_143_r2.png', 'path': 'imgs/blob_143_r2.png'},
    {'name': 'imgs/blob_57_r1.png', 'path': 'imgs/blob_57_r1.png'},
    {'name': 'imgs/blob_242.png', 'path': 'imgs/blob_242.png'},
    {'name': 'imgs/blob_88_r1.png', 'path': 'imgs/blob_88_r1.png'},
    {'name': 'imgs/blob_119_r3.png', 'path': 'imgs/blob_119_r3.png'},
    {'name': 'imgs/blob_100_r1.png', 'path': 'imgs/blob_100_r1.png'},
    {'name': 'imgs/blob_250_r2.png', 'path': 'imgs/blob_250_r2.png'},
    {'name': 'imgs/blob_152.png', 'path': 'imgs/blob_152.png'},
    {'name': 'imgs/blob_347_r2.png', 'path': 'imgs/blob_347_r2.png'},
    {'name': 'imgs/blob_35_r3.png', 'path': 'imgs/blob_35_r3.png'},
    {'name': 'imgs/blob_311.png', 'path': 'imgs/blob_311.png'},
    {'name': 'imgs/blob_310_r3.png', 'path': 'imgs/blob_310_r3.png'},
    {'name': 'imgs/blob_257_r2.png', 'path': 'imgs/blob_257_r2.png'},
    {'name': 'imgs/blob_271.png', 'path': 'imgs/blob_271.png'},
    {'name': 'imgs/blob_260_r1.png', 'path': 'imgs/blob_260_r1.png'},
    {'name': 'imgs/blob_405.png', 'path': 'imgs/blob_405.png'},
    {'name': 'imgs/blob_144.png', 'path': 'imgs/blob_144.png'},
    {'name': 'imgs/blob_49_r2.png', 'path': 'imgs/blob_49_r2.png'},
    {'name': 'imgs/blob_301.png', 'path': 'imgs/blob_301.png'},
    {'name': 'imgs/blob_80.png', 'path': 'imgs/blob_80.png'},
    {'name': 'imgs/blob_260.png', 'path': 'imgs/blob_260.png'},
    {'name': 'imgs/blob_109_r1.png', 'path': 'imgs/blob_109_r1.png'},
    {'name': 'imgs/blob_77.png', 'path': 'imgs/blob_77.png'},
    {'name': 'imgs/blob_153_r3.png', 'path': 'imgs/blob_153_r3.png'},
    {'name': 'imgs/blob_324_r2.png', 'path': 'imgs/blob_324_r2.png'},
    {'name': 'imgs/blob_180_r1.png', 'path': 'imgs/blob_180_r1.png'},
    {'name': 'imgs/blob_9_r1.png', 'path': 'imgs/blob_9_r1.png'},
    {'name': 'imgs/blob_204_r3.png', 'path': 'imgs/blob_204_r3.png'},
    {'name': 'imgs/blob_215_r3.png', 'path': 'imgs/blob_215_r3.png'},
    {'name': 'imgs/blob_247_r2.png', 'path': 'imgs/blob_247_r2.png'},
    {'name': 'imgs/blob_347_r1.png', 'path': 'imgs/blob_347_r1.png'},
    {'name': 'imgs/blob_135_r3.png', 'path': 'imgs/blob_135_r3.png'},
    {'name': 'imgs/blob_402.png', 'path': 'imgs/blob_402.png'},
    {'name': 'imgs/blob_294_r2.png', 'path': 'imgs/blob_294_r2.png'},
    {'name': 'imgs/blob_69_r3.png', 'path': 'imgs/blob_69_r3.png'},
    {'name': 'imgs/blob_185.png', 'path': 'imgs/blob_185.png'},
    {'name': 'imgs/blob_129_r1.png', 'path': 'imgs/blob_129_r1.png'},
    {'name': 'imgs/blob_4_r1.png', 'path': 'imgs/blob_4_r1.png'}
  ]
});

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.DEBUG);


var frameDur;
async function updateInfo() {
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2022.1.4';
  expInfo['OS'] = window.navigator.platform;

  psychoJS.experiment.dataFileName = (("." + "/") + ("mainbright_" + expInfo["date"]));

  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  
  return Scheduler.Event.NEXT;
}


var setupClock;
var group;
var deadline_test;
var browser_check;
var userAgent;
var opera;
var imgs;
var resources;
var browser_outClock;
var out;
var continue_info_out;
var continue_border_out;
var scalingClock;
var header_border_scaling;
var header_text_scaling;
var show_keys;
var show_touch;
var oldt;
var x_size;
var y_size;
var screen_height;
var x_scale;
var y_scale;
var dbase;
var unittext;
var vsize;
var text_top;
var ccimage;
var continue_info_scaling;
var continue_border_scaling;
var premature_endClock;
var end;
var continue_info_end;
var continue_border_end;
var instruction1Clock;
var header_border1;
var header_text1;
var info1;
var continue_info1;
var continue_border1;
var key_resp1;
var instruction2Clock;
var header_border2;
var header_text2;
var info2;
var continue_info2;
var continue_border2;
var key_resp2;
var prepClock;
var group_instruction1;
var group_instruction3;
var countdownClock;
var trial_instruction_text1;
var a_border1;
var l_border1;
var label_a1;
var label_l1;
var a_key1;
var l_key1;
var count5;
var count4;
var count3;
var count2;
var count1;
var fixation_crossClock;
var trial_instruction_text_cross;
var a_border_cross;
var l_border_cross;
var label_a_cross;
var label_l_cross;
var a_key_cross;
var l_key_cross;
var fixation_cross_text;
var rt_trialClock;
var image_rt;
var trial_instruction_text_rt;
var key_resp;
var a_border_rt;
var l_border_rt;
var label_a_rt;
var label_l_rt;
var a_key_rt;
var l_key_rt;
var feedbackClock;
var trial_instruction_text_fb;
var a_border_fb;
var l_border_fb;
var label_a_fb;
var label_l_fb;
var a_key_fb;
var l_key_fb;
var text_fb;
var blank_screenClock;
var trial_instruction_text_ISI;
var a_border3;
var l_border3;
var label_a3;
var label_l3;
var a_key3;
var l_key3;
var instruction3Clock;
var header_border3;
var header_text3;
var info3;
var continue_info3;
var continue_border3;
var key_resp3;
var instruction4Clock;
var header_border4;
var header_text4;
var continue_info4;
var continue_border4;
var key_resp4;
var info4;
var InterimClock;
var header_border_interim;
var header_text_interim;
var text_interim;
var continue_interim;
var continue_border_interim;
var key_resp_interim;
var da;
var date_start;
var globalClock;
var routineTimer;
async function experimentInit() {
  // Initialize components for Routine "setup"
  setupClock = new util.Clock();
  //Switch keys
  /*
  if ((parseInt(expInfo["pp"])) % 2){
      condition = ["Dunkel", "Hell"];
      adjective = ["dunkler", "heller"];
      key_assigned = ["L", "A"];
      hell = "l"; //key_assigned_bright
      dunkel = "a"; //key_assigned_dark
  }
  
  condition = ["Dunkel", "Hell"];
  adjective = ["dunkler", "heller"];
  key_assigned = ["L", "A"];
  hell = "l"; //key_assigned_bright
  dunkel = "a"; //key_assigned_dark
  */
  
  //Assign to group
  const sel_group = Math.floor(Math.random() * 3);
  group = groups[sel_group]
  if (group != "control") {
      deadline_test = .9;
  }
  
  browser_check = isBrowserUpToDate();
  userAgent = navigator.userAgent;
  opera = (navigator.userAgent.match(/Opera|OPR\//) ? true : false);
  
  imgs = [
    {
      "imgs": ["imgs/blob_144.png", "imgs/blob_301.png", "imgs/blob_114.png", "imgs/blob_249.png", "imgs/blob_131.png", "imgs/blob_58.png", "imgs/blob_381.png", "imgs/blob_211.png", "imgs/blob_166.png", "imgs/blob_98.png", "imgs/blob_212.png", "imgs/blob_74.png", "imgs/blob_321.png", "imgs/blob_124.png", "imgs/blob_218.png", "imgs/blob_320.png"],
      "cutoff": [0.515, 0.485, 0.515, 0.485, 0.515, 0.515, 0.485, 0.485, 0.515, 0.515, 0.485, 0.515, 0.485, 0.515, 0.485, 0.485],
      "brightness": ["bright", "dark", "bright", "dark", "bright", "bright", "dark", "dark", "bright", "bright", "dark", "bright", "dark", "bright", "dark", "dark"],
      "trial_type": ["practice", "practice", "practice", "practice", "practice", "practice", "practice", "practice", "practice", "practice", "practice", "practice", "practice", "practice", "practice", "practice"],
      "rotate": ["NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"],
      "block": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      "size_x": [750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750],
      "size_y": [750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750],
      "trial_nr": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    },
    {
      "imgs": ["imgs/blob_18.png", "imgs/blob_296.png", "imgs/blob_289.png", "imgs/blob_302.png", "imgs/blob_13.png", "imgs/blob_153.png", "imgs/blob_188.png", "imgs/blob_30.png", "imgs/blob_311.png", "imgs/blob_239.png", "imgs/blob_66.png", "imgs/blob_287.png", "imgs/blob_369.png", "imgs/blob_202.png", "imgs/blob_35.png", "imgs/blob_193.png", "imgs/blob_396.png", "imgs/blob_313.png", "imgs/blob_295.png", "imgs/blob_312.png", "imgs/blob_143.png", "imgs/blob_359.png", "imgs/blob_0.png", "imgs/blob_89.png", "imgs/blob_49.png", "imgs/blob_171.png", "imgs/blob_275.png", "imgs/blob_242.png", "imgs/blob_207.png", "imgs/blob_371.png", "imgs/blob_57.png", "imgs/blob_406.png", "imgs/blob_151.png", "imgs/blob_204.png", "imgs/blob_324.png", "imgs/blob_257.png", "imgs/blob_250.png", "imgs/blob_129.png", "imgs/blob_3.png", "imgs/blob_88.png", "imgs/blob_260.png", "imgs/blob_347.png", "imgs/blob_135.png", "imgs/blob_69.png", "imgs/blob_378.png", "imgs/blob_215.png", "imgs/blob_185.png", "imgs/blob_267.png", "imgs/blob_278.png", "imgs/blob_362.png", "imgs/blob_391.png", "imgs/blob_152.png", "imgs/blob_280.png", "imgs/blob_9.png", "imgs/blob_137.png", "imgs/blob_189.png", "imgs/blob_15.png", "imgs/blob_12.png", "imgs/blob_87.png", "imgs/blob_80.png", "imgs/blob_187.png", "imgs/blob_310.png", "imgs/blob_100.png", "imgs/blob_247.png", "imgs/blob_339.png", "imgs/blob_322.png", "imgs/blob_125.png", "imgs/blob_269.png", "imgs/blob_77.png", "imgs/blob_271.png", "imgs/blob_254.png", "imgs/blob_109.png", "imgs/blob_394.png", "imgs/blob_308.png", "imgs/blob_355.png", "imgs/blob_67.png", "imgs/blob_91.png", "imgs/blob_262.png", "imgs/blob_345.png", "imgs/blob_107.png", "imgs/blob_165.png", "imgs/blob_56.png", "imgs/blob_363.png", "imgs/blob_294.png", "imgs/blob_400.png", "imgs/blob_72.png", "imgs/blob_1.png", "imgs/blob_180.png", "imgs/blob_279.png", "imgs/blob_119.png", "imgs/blob_7.png", "imgs/blob_4.png", "imgs/blob_263.png", "imgs/blob_37.png"],
      "cutoff": [0.515, 0.485, 0.485, 0.485, 0.515, 0.515, 0.515, 0.515, 0.485, 0.485, 0.515, 0.485, 0.485, 0.485, 0.515, 0.485, 0.25, 0.485, 0.485, 0.485, 0.515, 0.485, 0.515, 0.515, 0.515, 0.515, 0.485, 0.485, 0.485, 0.485, 0.515, 0.75, 0.515, 0.485, 0.485, 0.485, 0.485, 0.515, 0.515, 0.515, 0.485, 0.485, 0.515, 0.515, 0.485, 0.485, 0.515, 0.485, 0.485, 0.485, 0.25, 0.515, 0.485, 0.515, 0.515, 0.515, 0.515, 0.515, 0.515, 0.515, 0.515, 0.485, 0.515, 0.485, 0.485, 0.485, 0.515, 0.485, 0.515, 0.485, 0.485, 0.515, 0.25, 0.485, 0.485, 0.515, 0.515, 0.485, 0.485, 0.515, 0.515, 0.515, 0.485, 0.485, 0.75, 0.515, 0.515, 0.515, 0.485, 0.515, 0.515, 0.515, 0.485, 0.515],
      "brightness": ["bright", "dark", "dark", "dark", "bright", "bright", "bright", "bright", "dark", "dark", "bright", "dark", "dark", "dark", "bright", "dark", "dark", "dark", "dark", "dark", "bright", "dark", "bright", "bright", "bright", "bright", "dark", "dark", "dark", "dark", "bright", "bright", "bright", "dark", "dark", "dark", "dark", "bright", "bright", "bright", "dark", "dark", "bright", "bright", "dark", "dark", "bright", "dark", "dark", "dark", "dark", "bright", "dark", "bright", "bright", "bright", "bright", "bright", "bright", "bright", "bright", "dark", "bright", "dark", "dark", "dark", "bright", "dark", "bright", "dark", "dark", "bright", "dark", "dark", "dark", "bright", "bright", "dark", "dark", "bright", "bright", "bright", "dark", "dark", "bright", "bright", "bright", "bright", "dark", "bright", "bright", "bright", "dark", "bright"],
      "trial_type": ["warmup", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test"],
      "rotate": ["NA", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "NA", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "NA", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "NA", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "NA", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "NA", 0, 0, 0, 0, 0, 0, 0, 0, 0],
      "block": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      "size_x": [750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750],
      "size_y": [750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750],
      "trial_nr": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94]
    },
    {
      "imgs": ["imgs/blob_366.png", "imgs/blob_296_r1.png", "imgs/blob_289_r1.png", "imgs/blob_302_r1.png", "imgs/blob_13_r1.png", "imgs/blob_153_r1.png", "imgs/blob_188_r1.png", "imgs/blob_30_r1.png", "imgs/blob_311_r1.png", "imgs/blob_239_r1.png", "imgs/blob_66_r1.png", "imgs/blob_287_r1.png", "imgs/blob_369_r1.png", "imgs/blob_392.png", "imgs/blob_202_r1.png", "imgs/blob_35_r1.png", "imgs/blob_193_r1.png", "imgs/blob_313_r1.png", "imgs/blob_295_r1.png", "imgs/blob_312_r1.png", "imgs/blob_143_r1.png", "imgs/blob_359_r1.png", "imgs/blob_0_r1.png", "imgs/blob_89_r1.png", "imgs/blob_49_r1.png", "imgs/blob_171_r1.png", "imgs/blob_275_r1.png", "imgs/blob_242_r1.png", "imgs/blob_207_r1.png", "imgs/blob_371_r1.png", "imgs/blob_57_r1.png", "imgs/blob_151_r1.png", "imgs/blob_204_r1.png", "imgs/blob_324_r1.png", "imgs/blob_257_r1.png", "imgs/blob_250_r1.png", "imgs/blob_129_r1.png", "imgs/blob_393.png", "imgs/blob_3_r1.png", "imgs/blob_88_r1.png", "imgs/blob_260_r1.png", "imgs/blob_347_r1.png", "imgs/blob_135_r1.png", "imgs/blob_69_r1.png", "imgs/blob_378_r1.png", "imgs/blob_215_r1.png", "imgs/blob_185_r1.png", "imgs/blob_267_r1.png", "imgs/blob_278_r1.png", "imgs/blob_398.png", "imgs/blob_362_r1.png", "imgs/blob_152_r1.png", "imgs/blob_280_r1.png", "imgs/blob_9_r1.png", "imgs/blob_137_r1.png", "imgs/blob_189_r1.png", "imgs/blob_15_r1.png", "imgs/blob_12_r1.png", "imgs/blob_87_r1.png", "imgs/blob_80_r1.png", "imgs/blob_187_r1.png", "imgs/blob_310_r1.png", "imgs/blob_100_r1.png", "imgs/blob_247_r1.png", "imgs/blob_339_r1.png", "imgs/blob_322_r1.png", "imgs/blob_125_r1.png", "imgs/blob_269_r1.png", "imgs/blob_77_r1.png", "imgs/blob_271_r1.png", "imgs/blob_254_r1.png", "imgs/blob_109_r1.png", "imgs/blob_308_r1.png", "imgs/blob_355_r1.png", "imgs/blob_67_r1.png", "imgs/blob_407.png", "imgs/blob_91_r1.png", "imgs/blob_262_r1.png", "imgs/blob_345_r1.png", "imgs/blob_107_r1.png", "imgs/blob_165_r1.png", "imgs/blob_56_r1.png", "imgs/blob_363_r1.png", "imgs/blob_294_r1.png", "imgs/blob_72_r1.png", "imgs/blob_1_r1.png", "imgs/blob_180_r1.png", "imgs/blob_279_r1.png", "imgs/blob_119_r1.png", "imgs/blob_7_r1.png", "imgs/blob_4_r1.png", "imgs/blob_263_r1.png", "imgs/blob_37_r1.png", "imgs/blob_404.png"],
      "cutoff": [0.485, 0.485, 0.485, 0.485, 0.515, 0.515, 0.515, 0.515, 0.485, 0.485, 0.515, 0.485, 0.485, 0.25, 0.485, 0.515, 0.485, 0.485, 0.485, 0.485, 0.515, 0.485, 0.515, 0.515, 0.515, 0.515, 0.485, 0.485, 0.485, 0.485, 0.515, 0.515, 0.485, 0.485, 0.485, 0.485, 0.515, 0.25, 0.515, 0.515, 0.485, 0.485, 0.515, 0.515, 0.485, 0.485, 0.515, 0.485, 0.485, 0.75, 0.485, 0.515, 0.485, 0.515, 0.515, 0.515, 0.515, 0.515, 0.515, 0.515, 0.515, 0.485, 0.515, 0.485, 0.485, 0.485, 0.515, 0.485, 0.515, 0.485, 0.485, 0.515, 0.485, 0.485, 0.515, 0.75, 0.515, 0.485, 0.485, 0.515, 0.515, 0.515, 0.485, 0.485, 0.515, 0.515, 0.515, 0.485, 0.515, 0.515, 0.515, 0.485, 0.515, 0.75],
      "brightness": ["dark", "dark", "dark", "dark", "bright", "bright", "bright", "bright", "dark", "dark", "bright", "dark", "dark", "dark", "dark", "bright", "dark", "dark", "dark", "dark", "bright", "dark", "bright", "bright", "bright", "bright", "dark", "dark", "dark", "dark", "bright", "bright", "dark", "dark", "dark", "dark", "bright", "dark", "bright", "bright", "dark", "dark", "bright", "bright", "dark", "dark", "bright", "dark", "dark", "bright", "dark", "bright", "dark", "bright", "bright", "bright", "bright", "bright", "bright", "bright", "bright", "dark", "bright", "dark", "dark", "dark", "bright", "dark", "bright", "dark", "dark", "bright", "dark", "dark", "bright", "bright", "bright", "dark", "dark", "bright", "bright", "bright", "dark", "dark", "bright", "bright", "bright", "dark", "bright", "bright", "bright", "dark", "bright", "bright"],
      "trial_type": ["warmup", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention"],
      "rotate": ["NA", 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, "NA", 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, "NA", 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, "NA", 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, "NA", 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, "NA"],
      "block": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      "size_x": [750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750],
      "size_y": [750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750],
      "trial_nr": [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188]
    },
    {
      "imgs": ["imgs/blob_377.png", "imgs/blob_296_r2.png", "imgs/blob_289_r2.png", "imgs/blob_302_r2.png", "imgs/blob_13_r2.png", "imgs/blob_153_r2.png", "imgs/blob_188_r2.png", "imgs/blob_30_r2.png", "imgs/blob_311_r2.png", "imgs/blob_239_r2.png", "imgs/blob_66_r2.png", "imgs/blob_287_r2.png", "imgs/blob_369_r2.png", "imgs/blob_202_r2.png", "imgs/blob_35_r2.png", "imgs/blob_193_r2.png", "imgs/blob_313_r2.png", "imgs/blob_295_r2.png", "imgs/blob_402.png", "imgs/blob_312_r2.png", "imgs/blob_143_r2.png", "imgs/blob_359_r2.png", "imgs/blob_0_r2.png", "imgs/blob_89_r2.png", "imgs/blob_49_r2.png", "imgs/blob_171_r2.png", "imgs/blob_275_r2.png", "imgs/blob_242_r2.png", "imgs/blob_401.png", "imgs/blob_207_r2.png", "imgs/blob_371_r2.png", "imgs/blob_57_r2.png", "imgs/blob_151_r2.png", "imgs/blob_204_r2.png", "imgs/blob_324_r2.png", "imgs/blob_257_r2.png", "imgs/blob_250_r2.png", "imgs/blob_129_r2.png", "imgs/blob_3_r2.png", "imgs/blob_88_r2.png", "imgs/blob_260_r2.png", "imgs/blob_347_r2.png", "imgs/blob_135_r2.png", "imgs/blob_69_r2.png", "imgs/blob_378_r2.png", "imgs/blob_215_r2.png", "imgs/blob_185_r2.png", "imgs/blob_267_r2.png", "imgs/blob_278_r2.png", "imgs/blob_362_r2.png", "imgs/blob_152_r2.png", "imgs/blob_280_r2.png", "imgs/blob_9_r2.png", "imgs/blob_389.png", "imgs/blob_137_r2.png", "imgs/blob_189_r2.png", "imgs/blob_15_r2.png", "imgs/blob_12_r2.png", "imgs/blob_87_r2.png", "imgs/blob_80_r2.png", "imgs/blob_187_r2.png", "imgs/blob_310_r2.png", "imgs/blob_100_r2.png", "imgs/blob_247_r2.png", "imgs/blob_339_r2.png", "imgs/blob_322_r2.png", "imgs/blob_125_r2.png", "imgs/blob_269_r2.png", "imgs/blob_77_r2.png", "imgs/blob_271_r2.png", "imgs/blob_254_r2.png", "imgs/blob_109_r2.png", "imgs/blob_308_r2.png", "imgs/blob_397.png", "imgs/blob_355_r2.png", "imgs/blob_67_r2.png", "imgs/blob_91_r2.png", "imgs/blob_262_r2.png", "imgs/blob_345_r2.png", "imgs/blob_107_r2.png", "imgs/blob_165_r2.png", "imgs/blob_56_r2.png", "imgs/blob_363_r2.png", "imgs/blob_294_r2.png", "imgs/blob_72_r2.png", "imgs/blob_1_r2.png", "imgs/blob_180_r2.png", "imgs/blob_388.png", "imgs/blob_279_r2.png", "imgs/blob_119_r2.png", "imgs/blob_7_r2.png", "imgs/blob_4_r2.png", "imgs/blob_263_r2.png", "imgs/blob_37_r2.png"],
      "cutoff": [0.485, 0.485, 0.485, 0.485, 0.515, 0.515, 0.515, 0.515, 0.485, 0.485, 0.515, 0.485, 0.485, 0.485, 0.515, 0.485, 0.485, 0.485, 0.75, 0.485, 0.515, 0.485, 0.515, 0.515, 0.515, 0.515, 0.485, 0.485, 0.75, 0.485, 0.485, 0.515, 0.515, 0.485, 0.485, 0.485, 0.485, 0.515, 0.515, 0.515, 0.485, 0.485, 0.515, 0.515, 0.485, 0.485, 0.515, 0.485, 0.485, 0.485, 0.515, 0.485, 0.515, 0.25, 0.515, 0.515, 0.515, 0.515, 0.515, 0.515, 0.515, 0.485, 0.515, 0.485, 0.485, 0.485, 0.515, 0.485, 0.515, 0.485, 0.485, 0.515, 0.485, 0.25, 0.485, 0.515, 0.515, 0.485, 0.485, 0.515, 0.515, 0.515, 0.485, 0.485, 0.515, 0.515, 0.515, 0.25, 0.485, 0.515, 0.515, 0.515, 0.485, 0.515],
      "brightness": ["dark", "dark", "dark", "dark", "bright", "bright", "bright", "bright", "dark", "dark", "bright", "dark", "dark", "dark", "bright", "dark", "dark", "dark", "bright", "dark", "bright", "dark", "bright", "bright", "bright", "bright", "dark", "dark", "bright", "dark", "dark", "bright", "bright", "dark", "dark", "dark", "dark", "bright", "bright", "bright", "dark", "dark", "bright", "bright", "dark", "dark", "bright", "dark", "dark", "dark", "bright", "dark", "bright", "dark", "bright", "bright", "bright", "bright", "bright", "bright", "bright", "dark", "bright", "dark", "dark", "dark", "bright", "dark", "bright", "dark", "dark", "bright", "dark", "dark", "dark", "bright", "bright", "dark", "dark", "bright", "bright", "bright", "dark", "dark", "bright", "bright", "bright", "dark", "dark", "bright", "bright", "bright", "dark", "bright"],
      "trial_type": ["warmup", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test"],
      "rotate": ["NA", 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, "NA", 180, 180, 180, 180, 180, 180, 180, 180, 180, "NA", 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, "NA", 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, "NA", 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, "NA", 180, 180, 180, 180, 180, 180],
      "block": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      "size_x": [750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750],
      "size_y": [750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750],
      "trial_nr": [189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282]
    },
    {
      "imgs": ["imgs/blob_103.png", "imgs/blob_296_r3.png", "imgs/blob_289_r3.png", "imgs/blob_302_r3.png", "imgs/blob_13_r3.png", "imgs/blob_153_r3.png", "imgs/blob_188_r3.png", "imgs/blob_30_r3.png", "imgs/blob_311_r3.png", "imgs/blob_399.png", "imgs/blob_239_r3.png", "imgs/blob_66_r3.png", "imgs/blob_287_r3.png", "imgs/blob_369_r3.png", "imgs/blob_202_r3.png", "imgs/blob_35_r3.png", "imgs/blob_193_r3.png", "imgs/blob_313_r3.png", "imgs/blob_295_r3.png", "imgs/blob_312_r3.png", "imgs/blob_143_r3.png", "imgs/blob_359_r3.png", "imgs/blob_0_r3.png", "imgs/blob_89_r3.png", "imgs/blob_49_r3.png", "imgs/blob_171_r3.png", "imgs/blob_275_r3.png", "imgs/blob_242_r3.png", "imgs/blob_207_r3.png", "imgs/blob_390.png", "imgs/blob_371_r3.png", "imgs/blob_57_r3.png", "imgs/blob_151_r3.png", "imgs/blob_204_r3.png", "imgs/blob_324_r3.png", "imgs/blob_257_r3.png", "imgs/blob_250_r3.png", "imgs/blob_129_r3.png", "imgs/blob_3_r3.png", "imgs/blob_88_r3.png", "imgs/blob_260_r3.png", "imgs/blob_347_r3.png", "imgs/blob_135_r3.png", "imgs/blob_69_r3.png", "imgs/blob_378_r3.png", "imgs/blob_215_r3.png", "imgs/blob_185_r3.png", "imgs/blob_267_r3.png", "imgs/blob_278_r3.png", "imgs/blob_362_r3.png", "imgs/blob_152_r3.png", "imgs/blob_280_r3.png", "imgs/blob_9_r3.png", "imgs/blob_137_r3.png", "imgs/blob_189_r3.png", "imgs/blob_15_r3.png", "imgs/blob_395.png", "imgs/blob_12_r3.png", "imgs/blob_87_r3.png", "imgs/blob_80_r3.png", "imgs/blob_187_r3.png", "imgs/blob_310_r3.png", "imgs/blob_100_r3.png", "imgs/blob_247_r3.png", "imgs/blob_339_r3.png", "imgs/blob_322_r3.png", "imgs/blob_125_r3.png", "imgs/blob_269_r3.png", "imgs/blob_77_r3.png", "imgs/blob_271_r3.png", "imgs/blob_254_r3.png", "imgs/blob_109_r3.png", "imgs/blob_308_r3.png", "imgs/blob_355_r3.png", "imgs/blob_403.png", "imgs/blob_67_r3.png", "imgs/blob_91_r3.png", "imgs/blob_262_r3.png", "imgs/blob_345_r3.png", "imgs/blob_107_r3.png", "imgs/blob_165_r3.png", "imgs/blob_56_r3.png", "imgs/blob_363_r3.png", "imgs/blob_294_r3.png", "imgs/blob_72_r3.png", "imgs/blob_1_r3.png", "imgs/blob_180_r3.png", "imgs/blob_279_r3.png", "imgs/blob_405.png", "imgs/blob_119_r3.png", "imgs/blob_7_r3.png", "imgs/blob_4_r3.png", "imgs/blob_263_r3.png", "imgs/blob_37_r3.png"],
      "cutoff": [0.515, 0.485, 0.485, 0.485, 0.515, 0.515, 0.515, 0.515, 0.485, 0.75, 0.485, 0.515, 0.485, 0.485, 0.485, 0.515, 0.485, 0.485, 0.485, 0.485, 0.515, 0.485, 0.515, 0.515, 0.515, 0.515, 0.485, 0.485, 0.485, 0.25, 0.485, 0.515, 0.515, 0.485, 0.485, 0.485, 0.485, 0.515, 0.515, 0.515, 0.485, 0.485, 0.515, 0.515, 0.485, 0.485, 0.515, 0.485, 0.485, 0.485, 0.515, 0.485, 0.515, 0.515, 0.515, 0.515, 0.25, 0.515, 0.515, 0.515, 0.515, 0.485, 0.515, 0.485, 0.485, 0.485, 0.515, 0.485, 0.515, 0.485, 0.485, 0.515, 0.485, 0.485, 0.75, 0.515, 0.515, 0.485, 0.485, 0.515, 0.515, 0.515, 0.485, 0.485, 0.515, 0.515, 0.515, 0.485, 0.75, 0.515, 0.515, 0.515, 0.485, 0.515],
      "brightness": ["bright", "dark", "dark", "dark", "bright", "bright", "bright", "bright", "dark", "bright", "dark", "bright", "dark", "dark", "dark", "bright", "dark", "dark", "dark", "dark", "bright", "dark", "bright", "bright", "bright", "bright", "dark", "dark", "dark", "dark", "dark", "bright", "bright", "dark", "dark", "dark", "dark", "bright", "bright", "bright", "dark", "dark", "bright", "bright", "dark", "dark", "bright", "dark", "dark", "dark", "bright", "dark", "bright", "bright", "bright", "bright", "dark", "bright", "bright", "bright", "bright", "dark", "bright", "dark", "dark", "dark", "bright", "dark", "bright", "dark", "dark", "bright", "dark", "dark", "bright", "bright", "bright", "dark", "dark", "bright", "bright", "bright", "dark", "dark", "bright", "bright", "bright", "dark", "bright", "bright", "bright", "bright", "dark", "bright"],
      "trial_type": ["warmup", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "test", "attention", "test", "test", "test", "test", "test"],
      "rotate": ["NA", 270, 270, 270, 270, 270, 270, 270, 270, "NA", 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, "NA", 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, "NA", 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, "NA", 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, "NA", 270, 270, 270, 270, 270],
      "block": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
      "size_x": [750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750],
      "size_y": [750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750],
      "trial_nr": [283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376]
    }
  ]
  resources = {
    status: PsychoJS.Status.NOT_STARTED
  };
  // Initialize components for Routine "browser_out"
  browser_outClock = new util.Clock();
  out = new visual.TextStim({
    win: psychoJS.window,
    name: 'out',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  continue_info_out = new visual.Rect ({
    win: psychoJS.window, name: 'continue_info_out', 
    width: [2, 0.07][0], height: [2, 0.07][1],
    ori: 0.0, pos: [0, (- 0.462)],
    lineWidth: 1.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    opacity: undefined, depth: -2, interpolate: true,
  });
  
  continue_border_out = new visual.TextStim({
    win: psychoJS.window,
    name: 'continue_border_out',
    text: 'Um die Studie in diesem Fenster zu schließen, drücken Sie bitte zwei Mal die Escape-Taste.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, (- 0.462)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -3.0 
  });
  
  // Initialize components for Routine "scaling"
  scalingClock = new util.Clock();
  header_border_scaling = new visual.Rect ({
    win: psychoJS.window, name: 'header_border_scaling', 
    width: [2, 0.1][0], height: [2, 0.1][1],
    ori: 0.0, pos: [0, 0.44],
    lineWidth: 2.0, lineColor: new util.Color('white'),
    fillColor: new util.Color([0.0, 0.0, 0.0]),
    opacity: undefined, depth: 0, interpolate: true,
  });
  
  header_text_scaling = new visual.TextStim({
    win: psychoJS.window,
    name: 'header_text_scaling',
    text: 'Bevor es losgeht...',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.44], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  show_keys = 0;
  show_touch = 0;
  if ((expInfo["device type"] === "Touch screen")) {
      show_touch = 1;
  } else {
      show_keys = 1;
  }
  oldt = 0;
  x_size = 8.56;
  y_size = 5.398;
  screen_height = 0;
  if ((psychoJS.window.units === "norm")) {
      x_scale = 0.05;
      y_scale = 0.1;
      dbase = 0.0001;
      unittext = " norm units";
      vsize = 2;
  } else {
      if ((psychoJS.window.units === "pix")) {
          x_scale = 60;
          y_scale = 40;
          dbase = 0.1;
          unittext = " pixels";
          vsize = psychoJS.window.size[1];
      } else {
          x_scale = 0.05;
          y_scale = 0.05;
          dbase = 0.0001;
          unittext = " height units";
          vsize = 1;
      }
  }
  
  text_top = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_top',
    text: 'Bitte passen Sie die Größe des Bilds an die Größe einer Zahlungskarte (z. B. Kredit- oder EC-Karte) an. \nDies dient dazu, dass bei allen Teilnehmenden das Experiment gleich groß dargestellt wird.\nHalten Sie nun also eine Zahlungskarte vor den Bildschirm und bedienen die Pfeiltasten Ihrer Tastatur, sodass die Größe des Bilds der Größe der Zahlungskarte gleicht. \n\n↑ Pfeil nach oben für höher \n↓ Pfeil nach unten für weniger hoch\n← Pfeil nach links für schmaler \n→ Pfeil nach rechts für breiter',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.155], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -3.0 
  });
  
  ccimage = new visual.ImageStim({
    win : psychoJS.window,
    name : 'ccimage', units : undefined, 
    image : 'rsc/bank-1300155_640.png', mask : undefined,
    ori : 0.0, pos : [0, (- 0.23)], size : [(x_size * x_scale), (y_size * y_scale)],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -4.0 
  });
  continue_info_scaling = new visual.Rect ({
    win: psychoJS.window, name: 'continue_info_scaling', 
    width: [2, 0.07][0], height: [2, 0.07][1],
    ori: 0.0, pos: [0, (- 0.462)],
    lineWidth: 1.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    opacity: undefined, depth: -5, interpolate: true,
  });
  
  continue_border_scaling = new visual.TextStim({
    win: psychoJS.window,
    name: 'continue_border_scaling',
    text: 'Bitte fahren Sie erst durch Drücken der Leertaste fort, wenn die Bild-Größe stimmt.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, (- 0.462)], height: 0.03,  wrapWidth: 5.0, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -6.0 
  });
  
  // Initialize components for Routine "premature_end"
  premature_endClock = new util.Clock();
  end = new visual.TextStim({
    win: psychoJS.window,
    name: 'end',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  continue_info_end = new visual.Rect ({
    win: psychoJS.window, name: 'continue_info_end', 
    width: [2, 0.07][0], height: [2, 0.07][1],
    ori: 0.0, pos: [0, (- 0.462)],
    lineWidth: 1.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    opacity: undefined, depth: -2, interpolate: true,
  });
  
  continue_border_end = new visual.TextStim({
    win: psychoJS.window,
    name: 'continue_border_end',
    text: 'Sie können das Browser-Fenster nun schließen.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, (- 0.462)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -3.0 
  });
  
  // Initialize components for Routine "instruction1"
  instruction1Clock = new util.Clock();
  header_border1 = new visual.Rect ({
    win: psychoJS.window, name: 'header_border1', 
    width: [2, 0.1][0], height: [2, 0.1][1],
    ori: 0.0, pos: [0, 0.44],
    lineWidth: 2.0, lineColor: new util.Color('white'),
    fillColor: new util.Color([0.0, 0.0, 0.0]),
    opacity: undefined, depth: 0, interpolate: true,
  });
  
  header_text1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'header_text1',
    text: 'Ihre Aufgabe',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.44], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  info1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'info1',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: -2.0 
  });
  
  continue_info1 = new visual.Rect ({
    win: psychoJS.window, name: 'continue_info1', 
    width: [2, 0.07][0], height: [2, 0.07][1],
    ori: 0.0, pos: [0, (- 0.462)],
    lineWidth: 1.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    opacity: undefined, depth: -3, interpolate: true,
  });
  
  continue_border1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'continue_border1',
    text: 'Um fortzufahren, drücken Sie bitte die Leertaste.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, (- 0.462)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -4.0 
  });
  
  key_resp1 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "instruction2"
  instruction2Clock = new util.Clock();
  header_border2 = new visual.Rect ({
    win: psychoJS.window, name: 'header_border2', 
    width: [2, 0.1][0], height: [2, 0.1][1],
    ori: 0.0, pos: [0, 0.44],
    lineWidth: 2.0, lineColor: new util.Color('white'),
    fillColor: new util.Color([0.0, 0.0, 0.0]),
    opacity: undefined, depth: 0, interpolate: true,
  });
  
  header_text2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'header_text2',
    text: 'Ihre Aufgabe',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.44], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  info2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'info2',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: -2.0 
  });
  
  continue_info2 = new visual.Rect ({
    win: psychoJS.window, name: 'continue_info2', 
    width: [2, 0.07][0], height: [2, 0.07][1],
    ori: 0.0, pos: [0, (- 0.462)],
    lineWidth: 1.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    opacity: undefined, depth: -3, interpolate: true,
  });
  
  continue_border2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'continue_border2',
    text: 'Um mit den Übungsdurchgängen zu beginnen, \ndrücken Sie bitte die Leertaste.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, (- 0.462)], height: 0.03,  wrapWidth: 3.0, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -4.0 
  });
  
  key_resp2 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "prep"
  prepClock = new util.Clock();
  if (group == "acc") {
      group_instruction1 = "Nun gilt zudem:" + "\n" + "Sie werden jeweils sehr wenig Zeit haben, um Ihre Entscheidung zu treffen. Bitte versuchen Sie, die Aufgabe trotz des Zeitdrucks gut zu bearbeiten: Es geht darum, möglichst wenige Fehler zu machen.";
      group_instruction3 = "wenn Sie zu langsam waren.";
  } else if (group == "gutfeeling") {
      group_instruction1 = "Nun gilt zudem:" + "\n" + "Sie werden jeweils sehr wenig Zeit haben, um Ihre Entscheidung zu treffen. Bitte versuchen Sie, die Aufgabe trotz des Zeitdrucks gut zu bearbeiten. Wenn Sie aber merken, dass Sie nicht rechtzeitig antworten können, folgen Sie Ihrem Bauchgefühl.";
      group_instruction3 = "wenn Sie zu langsam waren.";
  } else if (group == "control") {
      group_instruction1 = "Nun gilt zudem:" + "\n" + "Bitte versuchen Sie, die Aufgabe gut zu bearbeiten: Es geht darum, so schnell wie möglich zu antworten und gleichzeitig möglichst wenige Fehler zu machen.";
      group_instruction3 = "wenn Sie einen Fehler gemacht haben.";
  } else if (group == "gutfeeling2") {
      group_instruction1 = "Nun gilt zudem:" + "\n" + "Sie werden jeweils sehr wenig Zeit haben, um Ihre Entscheidung zu treffen. Bitte versuchen Sie, die Aufgabe trotz des Zeitdrucks gut zu bearbeiten. Wenn Sie sich aber unsicher sind, folgen Sie am besten Ihrem Bauchgefühl.";
      group_instruction3 = "wenn Sie zu langsam waren.";
  }
  // Initialize components for Routine "countdown"
  countdownClock = new util.Clock();
  trial_instruction_text1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'trial_instruction_text1',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, upper_text], height: textsize,  wrapWidth: 1.0, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  a_border1 = new visual.Rect ({
    win: psychoJS.window, name: 'a_border1', 
    width: [0.075, 0.075][0], height: [0.075, 0.075][1],
    ori: 0.0, pos: [(- 0.5), (- 0.44)],
    lineWidth: 2.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([0.3255, 0.3255, 0.3255]),
    opacity: undefined, depth: -1, interpolate: true,
  });
  
  l_border1 = new visual.Rect ({
    win: psychoJS.window, name: 'l_border1', 
    width: [0.075, 0.075][0], height: [0.075, 0.075][1],
    ori: 0.0, pos: [0.5, (- 0.44)],
    lineWidth: 2.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([0.3255, 0.3255, 0.3255]),
    opacity: undefined, depth: -2, interpolate: true,
  });
  
  label_a1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'label_a1',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [(- 0.5), (- 0.375)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -3.0 
  });
  
  label_l1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'label_l1',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0.5, (- 0.375)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -4.0 
  });
  
  a_key1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'a_key1',
    text: 'A',
    font: 'Open Sans',
    units: undefined, 
    pos: [(- 0.5), (- 0.44)], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -5.0 
  });
  
  l_key1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'l_key1',
    text: 'L',
    font: 'Open Sans',
    units: undefined, 
    pos: [0.5, (- 0.44)], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -6.0 
  });
  
  count5 = new visual.TextStim({
    win: psychoJS.window,
    name: 'count5',
    text: '•••••',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.2,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -7.0 
  });
  
  count4 = new visual.TextStim({
    win: psychoJS.window,
    name: 'count4',
    text: '••••',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.2,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -8.0 
  });
  
  count3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'count3',
    text: '•••',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.2,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -9.0 
  });
  
  count2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'count2',
    text: '••',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.2,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -10.0 
  });
  
  count1 = new visual.TextStim({
    win: psychoJS.window,
    name: 'count1',
    text: '•',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.2,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -11.0 
  });
  
  // Initialize components for Routine "fixation_cross"
  fixation_crossClock = new util.Clock();
  trial_instruction_text_cross = new visual.TextStim({
    win: psychoJS.window,
    name: 'trial_instruction_text_cross',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, upper_text], height: textsize,  wrapWidth: 1.0, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  a_border_cross = new visual.Rect ({
    win: psychoJS.window, name: 'a_border_cross', 
    width: [0.075, 0.075][0], height: [0.075, 0.075][1],
    ori: 0.0, pos: [(- 0.5), (- 0.44)],
    lineWidth: 2.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([0.3255, 0.3255, 0.3255]),
    opacity: undefined, depth: -1, interpolate: true,
  });
  
  l_border_cross = new visual.Rect ({
    win: psychoJS.window, name: 'l_border_cross', 
    width: [0.075, 0.075][0], height: [0.075, 0.075][1],
    ori: 0.0, pos: [0.5, (- 0.44)],
    lineWidth: 2.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([0.3255, 0.3255, 0.3255]),
    opacity: undefined, depth: -2, interpolate: true,
  });
  
  label_a_cross = new visual.TextStim({
    win: psychoJS.window,
    name: 'label_a_cross',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [(- 0.5), (- 0.375)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -3.0 
  });
  
  label_l_cross = new visual.TextStim({
    win: psychoJS.window,
    name: 'label_l_cross',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0.5, (- 0.375)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -4.0 
  });
  
  a_key_cross = new visual.TextStim({
    win: psychoJS.window,
    name: 'a_key_cross',
    text: 'A',
    font: 'Open Sans',
    units: undefined, 
    pos: [(- 0.5), (- 0.44)], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -5.0 
  });
  
  l_key_cross = new visual.TextStim({
    win: psychoJS.window,
    name: 'l_key_cross',
    text: 'L',
    font: 'Open Sans',
    units: undefined, 
    pos: [0.5, (- 0.44)], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -6.0 
  });
  
  fixation_cross_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'fixation_cross_text',
    text: '+',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.15,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -7.0 
  });
  
  // Initialize components for Routine "rt_trial"
  rt_trialClock = new util.Clock();
  image_rt = new visual.ImageStim({
    win : psychoJS.window,
    name : 'image_rt', units : undefined, 
    image : undefined, mask : undefined,
    ori : 0.0, pos : [0, 0], size : 1.0,
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : 0.0 
  });
  trial_instruction_text_rt = new visual.TextStim({
    win: psychoJS.window,
    name: 'trial_instruction_text_rt',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, upper_text], height: textsize,  wrapWidth: 1.0, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  key_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  a_border_rt = new visual.Rect ({
    win: psychoJS.window, name: 'a_border_rt', 
    width: [0.075, 0.075][0], height: [0.075, 0.075][1],
    ori: 0.0, pos: [(- 0.5), (- 0.44)],
    lineWidth: 2.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([0.3255, 0.3255, 0.3255]),
    opacity: undefined, depth: -3, interpolate: true,
  });
  
  l_border_rt = new visual.Rect ({
    win: psychoJS.window, name: 'l_border_rt', 
    width: [0.075, 0.075][0], height: [0.075, 0.075][1],
    ori: 0.0, pos: [0.5, (- 0.44)],
    lineWidth: 2.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([0.3255, 0.3255, 0.3255]),
    opacity: undefined, depth: -4, interpolate: true,
  });
  
  label_a_rt = new visual.TextStim({
    win: psychoJS.window,
    name: 'label_a_rt',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [(- 0.5), (- 0.375)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -5.0 
  });
  
  label_l_rt = new visual.TextStim({
    win: psychoJS.window,
    name: 'label_l_rt',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0.5, (- 0.375)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -6.0 
  });
  
  a_key_rt = new visual.TextStim({
    win: psychoJS.window,
    name: 'a_key_rt',
    text: 'A',
    font: 'Open Sans',
    units: undefined, 
    pos: [(- 0.5), (- 0.44)], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -7.0 
  });
  
  l_key_rt = new visual.TextStim({
    win: psychoJS.window,
    name: 'l_key_rt',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0.5, (- 0.44)], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -8.0 
  });
  
  var cutoff
  var cutoff_translated
  var trial_type
  // Initialize components for Routine "feedback"
  feedbackClock = new util.Clock();
  trial_instruction_text_fb = new visual.TextStim({
    win: psychoJS.window,
    name: 'trial_instruction_text_fb',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, upper_text], height: textsize,  wrapWidth: 1.0, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  a_border_fb = new visual.Rect ({
    win: psychoJS.window, name: 'a_border_fb', 
    width: [0.075, 0.075][0], height: [0.075, 0.075][1],
    ori: 0.0, pos: [(- 0.5), (- 0.44)],
    lineWidth: 2.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([0.3255, 0.3255, 0.3255]),
    opacity: undefined, depth: -1, interpolate: true,
  });
  
  l_border_fb = new visual.Rect ({
    win: psychoJS.window, name: 'l_border_fb', 
    width: [0.075, 0.075][0], height: [0.075, 0.075][1],
    ori: 0.0, pos: [0.5, (- 0.44)],
    lineWidth: 2.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([0.3255, 0.3255, 0.3255]),
    opacity: undefined, depth: -2, interpolate: true,
  });
  
  label_a_fb = new visual.TextStim({
    win: psychoJS.window,
    name: 'label_a_fb',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [(- 0.5), (- 0.375)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -3.0 
  });
  
  label_l_fb = new visual.TextStim({
    win: psychoJS.window,
    name: 'label_l_fb',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0.5, (- 0.375)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -4.0 
  });
  
  a_key_fb = new visual.TextStim({
    win: psychoJS.window,
    name: 'a_key_fb',
    text: 'A',
    font: 'Open Sans',
    units: undefined, 
    pos: [(- 0.5), (- 0.44)], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -5.0 
  });
  
  l_key_fb = new visual.TextStim({
    win: psychoJS.window,
    name: 'l_key_fb',
    text: 'L',
    font: 'Open Sans',
    units: undefined, 
    pos: [0.5, (- 0.44)], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -6.0 
  });
  
  text_fb = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_fb',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: -8.0 
  });
  
  // Initialize components for Routine "blank_screen"
  blank_screenClock = new util.Clock();
  trial_instruction_text_ISI = new visual.TextStim({
    win: psychoJS.window,
    name: 'trial_instruction_text_ISI',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, upper_text], height: textsize,  wrapWidth: 1.0, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  a_border3 = new visual.Rect ({
    win: psychoJS.window, name: 'a_border3', 
    width: [0.075, 0.075][0], height: [0.075, 0.075][1],
    ori: 0.0, pos: [(- 0.5), (- 0.44)],
    lineWidth: 2.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([0.3255, 0.3255, 0.3255]),
    opacity: undefined, depth: -1, interpolate: true,
  });
  
  l_border3 = new visual.Rect ({
    win: psychoJS.window, name: 'l_border3', 
    width: [0.075, 0.075][0], height: [0.075, 0.075][1],
    ori: 0.0, pos: [0.5, (- 0.44)],
    lineWidth: 2.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([0.3255, 0.3255, 0.3255]),
    opacity: undefined, depth: -2, interpolate: true,
  });
  
  label_a3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'label_a3',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [(- 0.5), (- 0.375)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -3.0 
  });
  
  label_l3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'label_l3',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0.5, (- 0.375)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -4.0 
  });
  
  a_key3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'a_key3',
    text: 'A',
    font: 'Open Sans',
    units: undefined, 
    pos: [(- 0.5), (- 0.44)], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -5.0 
  });
  
  l_key3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'l_key3',
    text: 'L',
    font: 'Open Sans',
    units: undefined, 
    pos: [0.5, (- 0.44)], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('black'),  opacity: undefined,
    depth: -6.0 
  });
  
  // Initialize components for Routine "instruction3"
  instruction3Clock = new util.Clock();
  header_border3 = new visual.Rect ({
    win: psychoJS.window, name: 'header_border3', 
    width: [2, 0.1][0], height: [2, 0.1][1],
    ori: 0.0, pos: [0, 0.44],
    lineWidth: 2.0, lineColor: new util.Color('white'),
    fillColor: new util.Color([0.0, 0.0, 0.0]),
    opacity: undefined, depth: 0, interpolate: true,
  });
  
  header_text3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'header_text3',
    text: 'Ihre Aufgabe',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.44], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  info3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'info3',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: -2.0 
  });
  
  continue_info3 = new visual.Rect ({
    win: psychoJS.window, name: 'continue_info3', 
    width: [2, 0.07][0], height: [2, 0.07][1],
    ori: 0.0, pos: [0, (- 0.462)],
    lineWidth: 1.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    opacity: undefined, depth: -3, interpolate: true,
  });
  
  continue_border3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'continue_border3',
    text: 'Um fortzufahren, drücken Sie bitte die Leertaste.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, (- 0.462)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -4.0 
  });
  
  key_resp3 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "instruction4"
  instruction4Clock = new util.Clock();
  header_border4 = new visual.Rect ({
    win: psychoJS.window, name: 'header_border4', 
    width: [2, 0.1][0], height: [2, 0.1][1],
    ori: 0.0, pos: [0, 0.44],
    lineWidth: 2.0, lineColor: new util.Color('white'),
    fillColor: new util.Color([0.0, 0.0, 0.0]),
    opacity: undefined, depth: 0, interpolate: true,
  });
  
  header_text4 = new visual.TextStim({
    win: psychoJS.window,
    name: 'header_text4',
    text: 'Ihre Aufgabe',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.44], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  continue_info4 = new visual.Rect ({
    win: psychoJS.window, name: 'continue_info4', 
    width: [2, 0.07][0], height: [2, 0.07][1],
    ori: 0.0, pos: [0, (- 0.462)],
    lineWidth: 1.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    opacity: undefined, depth: -2, interpolate: true,
  });
  
  continue_border4 = new visual.TextStim({
    win: psychoJS.window,
    name: 'continue_border4',
    text: 'Um mit dem 1. von 4 Test-Blöcken zu beginnen, \ndrücken Sie bitte die Leertaste.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, (- 0.462)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -3.0 
  });
  
  key_resp4 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  info4 = new visual.TextStim({
    win: psychoJS.window,
    name: 'info4',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: -5.0 
  });
  
  // Initialize components for Routine "Interim"
  InterimClock = new util.Clock();
  header_border_interim = new visual.Rect ({
    win: psychoJS.window, name: 'header_border_interim', 
    width: [2, 0.1][0], height: [2, 0.1][1],
    ori: 0.0, pos: [0, 0.44],
    lineWidth: 2.0, lineColor: new util.Color('white'),
    fillColor: new util.Color([0.0, 0.0, 0.0]),
    opacity: undefined, depth: 0, interpolate: true,
  });
  
  header_text_interim = new visual.TextStim({
    win: psychoJS.window,
    name: 'header_text_interim',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.44], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  text_interim = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_interim',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -2.0 
  });
  
  continue_interim = new visual.Rect ({
    win: psychoJS.window, name: 'continue_interim', 
    width: [2, 0.07][0], height: [2, 0.07][1],
    ori: 0.0, pos: [0, (- 0.462)],
    lineWidth: 1.0, lineColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    fillColor: new util.Color([(- 0.1461), (- 0.1461), (- 0.1461)]),
    opacity: undefined, depth: -3, interpolate: true,
  });
  
  continue_border_interim = new visual.TextStim({
    win: psychoJS.window,
    name: 'continue_border_interim',
    text: '',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, (- 0.462)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    color: new util.Color('white'),  opacity: undefined,
    depth: -4.0 
  });
  
  key_resp_interim = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  da = new Date();
  date_start = da.getFullYear() +"-"+ (da.getMonth()+1) +"-"+ da.getDate() +" "+ da.getHours() +":"+ da.getMinutes() +":"+ da.getSeconds();
  psychoJS.experiment.addData("date_start", date_start);
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}


var t;
var frameN;
var continueRoutine;
var setupComponents;
function setupRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //------Prepare to start Routine 'setup'-------
    t = 0;
    setupClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    // keep track of which components have finished
    setupComponents = [];
    setupComponents.push(resources);
    
    for (const thisComponent of setupComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function setupRoutineEachFrame() {
  return async function () {
    //------Loop for each frame of Routine 'setup'-------
    // get current time
    t = setupClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    if(resources.status == FINISHED){
        continueRoutine = False
    }    
    // start downloading resources specified by component resources
    if (t >= 0 && resources.status === PsychoJS.Status.NOT_STARTED) {
      console.log('register and start downloading resources specified by component resources');
      await psychoJS.serverManager.prepareResources(["rsc/bank-1300155_640.png"]);
      resources.status = PsychoJS.Status.STARTED;
    }
    // check on the resources specified by component resources
    if (t >= null && resources.status === PsychoJS.Status.STARTED) {
      if (psychoJS.serverManager.getResourceStatus(["rsc/bank-1300155_640.png"]) === core.ServerManager.ResourceStatus.DOWNLOADED) {
        console.log('finished downloading resources specified by component resources');
        resources.status = PsychoJS.Status.FINISHED;
      } else {
        console.log('resource specified in resources took longer than expected to download');
      }
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of setupComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function setupRoutineEnd() {
  return async function () {
    //------Ending Routine 'setup'-------
    for (const thisComponent of setupComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('key_bright', hell);
    psychoJS.experiment.addData('key_dark', dunkel);
    // the Routine "setup" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var text_out;
var text_out_time;
var browser_outComponents;
function browser_outRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //------Prepare to start Routine 'browser_out'-------
    t = 0;
    browser_outClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    if(browser_check && !safari && !opera){
        psychoJS.experiment.addData('browser', userAgent);
    }
    if(!browser_check) {
        text_out = 'Leider ist Ihr Browser (' + userAgent + ') nicht mehr aktuell.' + '\n\n' + 
        'Bitte updaten Sie Ihren Browser oder öffnen einen anderen, aktuelleren Browser.' + 
        '\n\n' + 'Um zur Studie zurückzukehren, nutzen Sie dann bitte den folgenden Link' + '\n' + 
        '(Tipp: Notieren Sie sich den Link jetzt):' + '\n' + 
        'https://pavlovia.org/psychmeth-uni-kiel/deadline_pretest?pp=' + expInfo["pp"]
        text_out_time = 999999
    }
    if(safari || opera) {
        text_out = 'Für die Studienteilnahme ist Mozilla Firefox, Microsoft Edge oder Google Chrome zu nutzen.' + 
        '\n\n' + 'Um teilzunehmen, öffnen Sie bitte einen der genannten Browser.' +
        '\n\n' + 'Nutzen Sie dann bitte den folgenden Link' + '\n' + 
        '(Tipp: Notieren Sie sich den Link jetzt):' + '\n' + 
        'https://pavlovia.org/psychmeth-uni-kiel/deadline_pretest?pp=' + expInfo["pp"]
        text_out_time = 999999
    }
    out.setText(text_out);
    // keep track of which components have finished
    browser_outComponents = [];
    browser_outComponents.push(out);
    browser_outComponents.push(continue_info_out);
    browser_outComponents.push(continue_border_out);
    
    for (const thisComponent of browser_outComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


var frameRemains;
function browser_outRoutineEachFrame() {
  return async function () {
    //------Loop for each frame of Routine 'browser_out'-------
    // get current time
    t = browser_outClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *out* updates
    if (t >= 0.0 && out.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      out.tStart = t;  // (not accounting for frame time here)
      out.frameNStart = frameN;  // exact frame index
      
      out.setAutoDraw(true);
    }

    frameRemains = 0.0 + text_out_time - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (out.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      out.setAutoDraw(false);
    }
    
    // *continue_info_out* updates
    if (t >= 0.0 && continue_info_out.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_info_out.tStart = t;  // (not accounting for frame time here)
      continue_info_out.frameNStart = frameN;  // exact frame index
      
      continue_info_out.setAutoDraw(true);
    }

    frameRemains = 0.0 + text_out_time - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (continue_info_out.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      continue_info_out.setAutoDraw(false);
    }
    
    // *continue_border_out* updates
    if (t >= 0.0 && continue_border_out.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_border_out.tStart = t;  // (not accounting for frame time here)
      continue_border_out.frameNStart = frameN;  // exact frame index
      
      continue_border_out.setAutoDraw(true);
    }

    frameRemains = 0.0 + text_out_time - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (continue_border_out.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      continue_border_out.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of browser_outComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function browser_outRoutineEnd() {
  return async function () {
    //------Ending Routine 'browser_out'-------
    for (const thisComponent of browser_outComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    if(!browser_check || safari || opera) {
        quitPsychoJS()
    }
    
    // the Routine "browser_out" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var scalingComponents;
function scalingRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //------Prepare to start Routine 'scaling'-------
    t = 0;
    scalingClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    console.log("key start");
    psychoJS.eventManager.clearEvents();
    
    // keep track of which components have finished
    scalingComponents = [];
    scalingComponents.push(header_border_scaling);
    scalingComponents.push(header_text_scaling);
    scalingComponents.push(text_top);
    scalingComponents.push(ccimage);
    scalingComponents.push(continue_info_scaling);
    scalingComponents.push(continue_border_scaling);
    
    for (const thisComponent of scalingComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


var _pj;
var keys;
var dscale;
function scalingRoutineEachFrame() {
  return async function () {
    //------Loop for each frame of Routine 'scaling'-------
    // get current time
    t = scalingClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *header_border_scaling* updates
    if (t >= 0.0 && header_border_scaling.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      header_border_scaling.tStart = t;  // (not accounting for frame time here)
      header_border_scaling.frameNStart = frameN;  // exact frame index
      
      header_border_scaling.setAutoDraw(true);
    }

    
    // *header_text_scaling* updates
    if (t >= 0.0 && header_text_scaling.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      header_text_scaling.tStart = t;  // (not accounting for frame time here)
      header_text_scaling.frameNStart = frameN;  // exact frame index
      
      header_text_scaling.setAutoDraw(true);
    }

    var _pj;
    function _pj_snippets(container) {
        function in_es6(left, right) {
            if (((right instanceof Array) || ((typeof right) === "string"))) {
                return (right.indexOf(left) > (- 1));
            } else {
                if (((right instanceof Map) || (right instanceof Set) || (right instanceof WeakMap) || (right instanceof WeakSet))) {
                    return right.has(left);
                } else {
                    return (left in right);
                }
            }
        }
        container["in_es6"] = in_es6;
        return container;
    }
    _pj = {};
    _pj_snippets(_pj);
    keys = psychoJS.eventManager.getKeys();
    if (keys.length) {
        if (((t - oldt) < 0.5)) {
            dscale = (5 * dbase);
            oldt = t;
        } else {
            dscale = dbase;
            oldt = t;
        }
        if ((_pj.in_es6("space", keys) && (t > 1))) {
            continueRoutine = false;
        } else {
            if (_pj.in_es6("up", keys)) {
                y_scale = (Math.round(((y_scale + dscale) * 10000)) / 10000);
            } else {
                if (_pj.in_es6("down", keys)) {
                    y_scale = (Math.round(((y_scale - dscale) * 10000)) / 10000);
                } else {
                    if (_pj.in_es6("left", keys)) {
                        x_scale = (Math.round(((x_scale - dscale) * 10000)) / 10000);
                    } else {
                        if (_pj.in_es6("right", keys)) {
                            x_scale = (Math.round(((x_scale + dscale) * 10000)) / 10000);
                        }
                    }
                }
            }
        }
        screen_height = (Math.round(((vsize * 10) / y_scale)) / 10);
        //text_bottom.text = (((((((("X Größe = " + x_scale.toString()) + unittext) + " pro cm, Y Scale = ") + y_scale.toString()) + unittext) + " per cm\nScreen height = ") + screen_height.toString()) + " cm\n\nPress the space bar when done");
        ccimage.size = [(x_size * x_scale), (y_size * y_scale)];
    }
    
    
    // *text_top* updates
    if (t >= 0.0 && text_top.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_top.tStart = t;  // (not accounting for frame time here)
      text_top.frameNStart = frameN;  // exact frame index
      
      text_top.setAutoDraw(true);
    }

    
    // *ccimage* updates
    if (t >= 0.0 && ccimage.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      ccimage.tStart = t;  // (not accounting for frame time here)
      ccimage.frameNStart = frameN;  // exact frame index
      
      ccimage.setAutoDraw(true);
    }

    
    // *continue_info_scaling* updates
    if (t >= 0.0 && continue_info_scaling.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_info_scaling.tStart = t;  // (not accounting for frame time here)
      continue_info_scaling.frameNStart = frameN;  // exact frame index
      
      continue_info_scaling.setAutoDraw(true);
    }

    
    // *continue_border_scaling* updates
    if (t >= 0.0 && continue_border_scaling.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_border_scaling.tStart = t;  // (not accounting for frame time here)
      continue_border_scaling.frameNStart = frameN;  // exact frame index
      
      continue_border_scaling.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of scalingComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function scalingRoutineEnd() {
  return async function () {
    //------Ending Routine 'scaling'-------
    for (const thisComponent of scalingComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData("X Scale", x_scale);
    psychoJS.experiment.addData("Y Scale", y_scale);
    // the Routine "scaling" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var text_end_time;
var premature_endComponents;
function premature_endRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //------Prepare to start Routine 'premature_end'-------
    t = 0;
    premature_endClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    if(isFullscreen() != 1){
        text_end += '\n\n' +
        'Es wurde festgestellt, dass Sie den Vollbildmodus verlassen haben. Deshalb ist eine Fortsetzung der Studie leider nicht möglich.' 
        text_end_time = 999999
    }
    /*
    if(condition_met_acc==0 && block_i==2) {
        text_end += '\n\n' +
        'Es liegen Anzeichen vor, dass Sie die Aufgabe nicht ernsthaft bearbeiten. Deshalb ist eine Fortsetzung der Studie leider nicht möglich.' 
        text_end_time = 999999
    }
    */
    end.setText(text_end);
    // keep track of which components have finished
    premature_endComponents = [];
    premature_endComponents.push(end);
    premature_endComponents.push(continue_info_end);
    premature_endComponents.push(continue_border_end);
    
    for (const thisComponent of premature_endComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function premature_endRoutineEachFrame() {
  return async function () {
    //------Loop for each frame of Routine 'premature_end'-------
    // get current time
    t = premature_endClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *end* updates
    if (t >= 0.0 && end.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      end.tStart = t;  // (not accounting for frame time here)
      end.frameNStart = frameN;  // exact frame index
      
      end.setAutoDraw(true);
    }

    frameRemains = 0.0 + text_end_time - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (end.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      end.setAutoDraw(false);
    }
    
    // *continue_info_end* updates
    if (t >= 0.0 && continue_info_end.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_info_end.tStart = t;  // (not accounting for frame time here)
      continue_info_end.frameNStart = frameN;  // exact frame index
      
      continue_info_end.setAutoDraw(true);
    }

    frameRemains = 0.0 + text_end_time - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (continue_info_end.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      continue_info_end.setAutoDraw(false);
    }
    
    // *continue_border_end* updates
    if (t >= 0.0 && continue_border_end.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_border_end.tStart = t;  // (not accounting for frame time here)
      continue_border_end.frameNStart = frameN;  // exact frame index
      
      continue_border_end.setAutoDraw(true);
    }

    frameRemains = 0.0 + text_end_time - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (continue_border_end.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      continue_border_end.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of premature_endComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function premature_endRoutineEnd() {
  return async function () {
    //------Ending Routine 'premature_end'-------
    for (const thisComponent of premature_endComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    if(text_end_time>0) {
        quitPsychoJS()
    }
    
    // the Routine "premature_end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp1_allKeys;
var instruction1Components;
function instruction1RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //------Prepare to start Routine 'instruction1'-------
    t = 0;
    instruction1Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    info1.setText((((((((((((((((((((((("Im Mittelpunkt der Studie steht eine Entscheidungsaufgabe:" + "\n") + "Ihnen werden in mehreren Durchg\u00e4ngen Quadrate mit hellen und dunklen Fl\u00e4chen (sog. Fleckenbilder) gezeigt. Ihre Aufgabe ist es, zu beurteilen, welche Fl\u00e4che insgesamt gr\u00f6\u00dfer ist.") + "\n\n") + "Wenn die Fl\u00e4che ") + adjective[0]) + " Flecken insgesamt gr\u00f6\u00dfer ist, dr\u00fccken Sie bitte die Taste \"A\" f\u00fcr die Antwort \"") + condition[0]) + "\".") + "\n") + "Wenn die Fl\u00e4che ") + adjective[1]) + " Flecken insgesamt gr\u00f6\u00dfer ist, dr\u00fccken Sie bitte die Taste \"L\" f\u00fcr die Antwort \"") + condition[1]) + "\".") + "\n\n") + "F\u00fcr die H\u00e4lfte aller Fleckenbilder ist \"") + condition[0]) + "\" die korrekte Antwort, f\u00fcr die andere H\u00e4lfte ist \"") + condition[1]) + "\" die korrekte Antwort.") + "\n\n") + "Vor der Pr\u00e4sentation jedes Fleckenbilds erscheint ein Kreuz in der Mitte des Bildschirms. Bitte richten Sie in jedem Durchgang Ihren Blick auf das Kreuz, bis das Fleckenbild erscheint."));
    key_resp1.keys = undefined;
    key_resp1.rt = undefined;
    _key_resp1_allKeys = [];
    // keep track of which components have finished
    instruction1Components = [];
    instruction1Components.push(header_border1);
    instruction1Components.push(header_text1);
    instruction1Components.push(info1);
    instruction1Components.push(continue_info1);
    instruction1Components.push(continue_border1);
    instruction1Components.push(key_resp1);
    
    for (const thisComponent of instruction1Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function instruction1RoutineEachFrame() {
  return async function () {
    //------Loop for each frame of Routine 'instruction1'-------
    // get current time
    t = instruction1Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *header_border1* updates
    if (t >= 0.0 && header_border1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      header_border1.tStart = t;  // (not accounting for frame time here)
      header_border1.frameNStart = frameN;  // exact frame index
      
      header_border1.setAutoDraw(true);
    }

    
    // *header_text1* updates
    if (t >= 0.0 && header_text1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      header_text1.tStart = t;  // (not accounting for frame time here)
      header_text1.frameNStart = frameN;  // exact frame index
      
      header_text1.setAutoDraw(true);
    }

    
    // *info1* updates
    if (t >= 0.0 && info1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      info1.tStart = t;  // (not accounting for frame time here)
      info1.frameNStart = frameN;  // exact frame index
      
      info1.setAutoDraw(true);
    }

    
    // *continue_info1* updates
    if (t >= 0.0 && continue_info1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_info1.tStart = t;  // (not accounting for frame time here)
      continue_info1.frameNStart = frameN;  // exact frame index
      
      continue_info1.setAutoDraw(true);
    }

    
    // *continue_border1* updates
    if (t >= 0.0 && continue_border1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_border1.tStart = t;  // (not accounting for frame time here)
      continue_border1.frameNStart = frameN;  // exact frame index
      
      continue_border1.setAutoDraw(true);
    }

    
    // *key_resp1* updates
    if (t >= 0.0 && key_resp1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp1.tStart = t;  // (not accounting for frame time here)
      key_resp1.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp1.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp1.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp1.clearEvents(); });
    }

    if (key_resp1.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp1.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp1_allKeys = _key_resp1_allKeys.concat(theseKeys);
      if (_key_resp1_allKeys.length > 0) {
        key_resp1.keys = _key_resp1_allKeys[_key_resp1_allKeys.length - 1].name;  // just the last key pressed
        key_resp1.rt = _key_resp1_allKeys[_key_resp1_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instruction1Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instruction1RoutineEnd() {
  return async function () {
    //------Ending Routine 'instruction1'-------
    for (const thisComponent of instruction1Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // update the trial handler
    if (psychoJS.experiment.currentLoop instanceof MultiStairHandler) {
      psychoJS.experiment.currentLoop.addResponse(key_resp1.corr, level);
    }
    psychoJS.experiment.addData('key_resp1.keys', key_resp1.keys);
    if (typeof key_resp1.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp1.rt', key_resp1.rt);
        routineTimer.reset();
        }
    
    key_resp1.stop();
    // the Routine "instruction1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp2_allKeys;
var instruction2Components;
function instruction2RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //------Prepare to start Routine 'instruction2'-------
    t = 0;
    instruction2Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    info2.setText((((((((((((("Legen Sie nun bitte Ihre beiden Zeigefinger auf den Tasten \"A\" und \"L\" ab." + "\n\n") + "Es folgen nun ein paar \u00dcbungsdurchg\u00e4nge.\nEs ist wichtig, dass Sie Ihre beiden Zeigefinger w\u00e4hren aller Durchg\u00e4nge auf diesen Tasten liegen lassen.") + "\n\n") + "Zur Erinnerung: Dr\u00fccken Sie die Taste \"A\" f\u00fcr \"") + condition[0]) + "\" (d. h., die Fl\u00e4che ") + adjective[0]) + " Flecken ist insgesamt gr\u00f6\u00dfer) und die Taste \"L\" f\u00fcr \"") + condition[1]) + "\" (d. h., die Fl\u00e4che ") + adjective[1]) + " Flecken ist insgesamt gr\u00f6\u00dfer)."));
    key_resp2.keys = undefined;
    key_resp2.rt = undefined;
    _key_resp2_allKeys = [];
    // keep track of which components have finished
    instruction2Components = [];
    instruction2Components.push(header_border2);
    instruction2Components.push(header_text2);
    instruction2Components.push(info2);
    instruction2Components.push(continue_info2);
    instruction2Components.push(continue_border2);
    instruction2Components.push(key_resp2);
    
    for (const thisComponent of instruction2Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function instruction2RoutineEachFrame() {
  return async function () {
    //------Loop for each frame of Routine 'instruction2'-------
    // get current time
    t = instruction2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *header_border2* updates
    if (t >= 0.0 && header_border2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      header_border2.tStart = t;  // (not accounting for frame time here)
      header_border2.frameNStart = frameN;  // exact frame index
      
      header_border2.setAutoDraw(true);
    }

    
    // *header_text2* updates
    if (t >= 0.0 && header_text2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      header_text2.tStart = t;  // (not accounting for frame time here)
      header_text2.frameNStart = frameN;  // exact frame index
      
      header_text2.setAutoDraw(true);
    }

    
    // *info2* updates
    if (t >= 0.0 && info2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      info2.tStart = t;  // (not accounting for frame time here)
      info2.frameNStart = frameN;  // exact frame index
      
      info2.setAutoDraw(true);
    }

    
    // *continue_info2* updates
    if (t >= 0.0 && continue_info2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_info2.tStart = t;  // (not accounting for frame time here)
      continue_info2.frameNStart = frameN;  // exact frame index
      
      continue_info2.setAutoDraw(true);
    }

    
    // *continue_border2* updates
    if (t >= 0.0 && continue_border2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_border2.tStart = t;  // (not accounting for frame time here)
      continue_border2.frameNStart = frameN;  // exact frame index
      
      continue_border2.setAutoDraw(true);
    }

    
    // *key_resp2* updates
    if (t >= 0.0 && key_resp2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp2.tStart = t;  // (not accounting for frame time here)
      key_resp2.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp2.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp2.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp2.clearEvents(); });
    }

    if (key_resp2.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp2.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp2_allKeys = _key_resp2_allKeys.concat(theseKeys);
      if (_key_resp2_allKeys.length > 0) {
        key_resp2.keys = _key_resp2_allKeys[_key_resp2_allKeys.length - 1].name;  // just the last key pressed
        key_resp2.rt = _key_resp2_allKeys[_key_resp2_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instruction2Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instruction2RoutineEnd() {
  return async function () {
    //------Ending Routine 'instruction2'-------
    for (const thisComponent of instruction2Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // update the trial handler
    if (psychoJS.experiment.currentLoop instanceof MultiStairHandler) {
      psychoJS.experiment.currentLoop.addResponse(key_resp2.corr, level);
    }
    psychoJS.experiment.addData('key_resp2.keys', key_resp2.keys);
    if (typeof key_resp2.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp2.rt', key_resp2.rt);
        routineTimer.reset();
        }
    
    key_resp2.stop();
    // the Routine "instruction2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var dat;
var date_within;
var block_images;
var block_cutoffs;
var trial_i;
var image_path;
var image_size;
var upper_text;
var interim_header;
var interim;
var interim_footer;
var deadline;
var group_instruction2;
var prepComponents;
function prepRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //------Prepare to start Routine 'prep'-------
    t = 0;
    prepClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    //Save date within
    dat = new Date();
    date_within = dat.getFullYear() +"-"+ (dat.getMonth()+1) +"-"+ dat.getDate() +" "+ dat.getHours() +":"+ dat.getMinutes() +":"+ dat.getSeconds();
    psychoJS.experiment.addData("date_within", date_within);
    
    block_i +=1
    block_images = imgs[block_i]['imgs'];
    block_cutoffs = imgs[block_i]['cutoff'];
    if(parseInt(expInfo["version"]) === 31415){
        block_images = [block_images[0]]
        block_images.push(imgs[block_i]['imgs'][1])
    }
    
    trial_i = 0
    image_path = block_images[trial_i]
    image_size = [12.5 * x_scale , 12.5 * y_scale] 
    upper_text = 7.5 * y_scale
    
    if(block_i < 4){
        interim_header = "Ihre Aufgabe"
        interim = "Sie haben nun den " + block_i + ". von 4 Test-Blöcken bearbeitet." + "\n\n" + "Gerne können Sie jetzt eine Pause machen." + "\n\n" + "Bitte verlassen Sie weiterhin NICHT den Vollbildmodus, d. h. auch NICHT während der Pause." 
        interim_footer = "Um fortzufahren, drücken Sie bitte die Leertaste."
    }else{
        interim_header = "Prima!"
         interim = "Sie haben nun auch die letzten Test-Durchgänge beendet." +  "\n\n" + "Zum Abschluss beantworten Sie bitte die nachfolgenden Fragen." + "\n\n" + 'Hierfür werden Sie auf eine neue Seite weitergeleitet, nachdem Sie die Leertaste gedrückt haben, abwarten, bis das Pop-up-Fenster mit der Nachricht "Closing the session. Please wait a few moments" verschwindet und beim darauffolgenden Pop-up-Fenster mit der Nachricht "Thank you for you patience" auf "Ok" geklickt haben.' + "\n\n" + "Bitte schließen Sie NICHT das Fenster!"
        interim_footer = "Um zu den Fragen weitergeleitet zu werden, drücken Sie bitte die Leertaste."
    }
    
    if(block_i != 0){
        deadline = deadline_test
        if (group == "acc") {
            group_instruction2 = "Welche Fläche ist insgesamt größer?" + "\n" + "Bitte versuchen Sie, die Aufgabe trotz des Zeitdrucks gut zu bearbeiten: Es geht darum, möglichst wenige Fehler zu machen.";
        } else if (group == "gutfeeling") {
            group_instruction2 = "Welche Fläche ist insgesamt größer?" + "\n" + "Bitte versuchen Sie, die Aufgabe trotz des Zeitdrucks gut zu bearbeiten. Wenn Sie aber merken, dass Sie nicht rechtzeitig antworten können, folgen Sie Ihrem Bauchgefühl.";
        } else if (group == "control") {
            group_instruction2 = "Welche Fläche ist insgesamt größer?" + "\n" + "Bitte versuchen Sie, die Aufgabe gut zu bearbeiten: Es geht darum, so schnell wie möglich zu antworten und gleichzeitig möglichst wenige Fehler zu machen.";
        } else if (group == "gutfeeling2") {
            group_instruction2 = "Welche Fläche ist insgesamt größer?" + "\n" + "Bitte versuchen Sie, die Aufgabe trotz des Zeitdrucks gut zu bearbeiten. Wenn Sie sich aber unsicher sind, folgen Sie am besten Ihrem Bauchgefühl.";
        }
    }
    // keep track of which components have finished
    prepComponents = [];
    
    for (const thisComponent of prepComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function prepRoutineEachFrame() {
  return async function () {
    //------Loop for each frame of Routine 'prep'-------
    // get current time
    t = prepClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of prepComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function prepRoutineEnd() {
  return async function () {
    //------Ending Routine 'prep'-------
    for (const thisComponent of prepComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "prep" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var countdownComponents;
function countdownRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //------Prepare to start Routine 'countdown'-------
    t = 0;
    countdownClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(5.000000);
    // update component parameters for each repeat
    trial_instruction_text1.setText(group_instruction2);
    label_a1.setText(condition[0]);
    label_l1.setText(condition[1]);
    // keep track of which components have finished
    countdownComponents = [];
    countdownComponents.push(trial_instruction_text1);
    countdownComponents.push(a_border1);
    countdownComponents.push(l_border1);
    countdownComponents.push(label_a1);
    countdownComponents.push(label_l1);
    countdownComponents.push(a_key1);
    countdownComponents.push(l_key1);
    countdownComponents.push(count5);
    countdownComponents.push(count4);
    countdownComponents.push(count3);
    countdownComponents.push(count2);
    countdownComponents.push(count1);
    
    for (const thisComponent of countdownComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function countdownRoutineEachFrame() {
  return async function () {
    //------Loop for each frame of Routine 'countdown'-------
    // get current time
    t = countdownClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *trial_instruction_text1* updates
    if (t >= 0.0 && trial_instruction_text1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      trial_instruction_text1.tStart = t;  // (not accounting for frame time here)
      trial_instruction_text1.frameNStart = frameN;  // exact frame index
      
      trial_instruction_text1.setAutoDraw(true);
    }

    frameRemains = 0.0 + 5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (trial_instruction_text1.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      trial_instruction_text1.setAutoDraw(false);
    }
    
    // *a_border1* updates
    if (t >= 0 && a_border1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      a_border1.tStart = t;  // (not accounting for frame time here)
      a_border1.frameNStart = frameN;  // exact frame index
      
      a_border1.setAutoDraw(true);
    }

    frameRemains = 0 + 5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (a_border1.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      a_border1.setAutoDraw(false);
    }
    
    // *l_border1* updates
    if (t >= 0.0 && l_border1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      l_border1.tStart = t;  // (not accounting for frame time here)
      l_border1.frameNStart = frameN;  // exact frame index
      
      l_border1.setAutoDraw(true);
    }

    frameRemains = 0.0 + 5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (l_border1.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      l_border1.setAutoDraw(false);
    }
    
    // *label_a1* updates
    if (t >= 0.0 && label_a1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      label_a1.tStart = t;  // (not accounting for frame time here)
      label_a1.frameNStart = frameN;  // exact frame index
      
      label_a1.setAutoDraw(true);
    }

    frameRemains = 0.0 + 5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (label_a1.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      label_a1.setAutoDraw(false);
    }
    
    // *label_l1* updates
    if (t >= 0.0 && label_l1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      label_l1.tStart = t;  // (not accounting for frame time here)
      label_l1.frameNStart = frameN;  // exact frame index
      
      label_l1.setAutoDraw(true);
    }

    frameRemains = 0.0 + 5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (label_l1.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      label_l1.setAutoDraw(false);
    }
    
    // *a_key1* updates
    if (t >= 0.0 && a_key1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      a_key1.tStart = t;  // (not accounting for frame time here)
      a_key1.frameNStart = frameN;  // exact frame index
      
      a_key1.setAutoDraw(true);
    }

    frameRemains = 0.0 + 5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (a_key1.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      a_key1.setAutoDraw(false);
    }
    
    // *l_key1* updates
    if (t >= 0.0 && l_key1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      l_key1.tStart = t;  // (not accounting for frame time here)
      l_key1.frameNStart = frameN;  // exact frame index
      
      l_key1.setAutoDraw(true);
    }

    frameRemains = 0.0 + 5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (l_key1.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      l_key1.setAutoDraw(false);
    }
    
    // *count5* updates
    if (t >= 0.0 && count5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      count5.tStart = t;  // (not accounting for frame time here)
      count5.frameNStart = frameN;  // exact frame index
      
      count5.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (count5.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      count5.setAutoDraw(false);
    }
    
    // *count4* updates
    if (t >= 1 && count4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      count4.tStart = t;  // (not accounting for frame time here)
      count4.frameNStart = frameN;  // exact frame index
      
      count4.setAutoDraw(true);
    }

    frameRemains = 1 + 1 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (count4.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      count4.setAutoDraw(false);
    }
    
    // *count3* updates
    if (t >= 2 && count3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      count3.tStart = t;  // (not accounting for frame time here)
      count3.frameNStart = frameN;  // exact frame index
      
      count3.setAutoDraw(true);
    }

    frameRemains = 2 + 1 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (count3.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      count3.setAutoDraw(false);
    }
    
    // *count2* updates
    if (t >= 3 && count2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      count2.tStart = t;  // (not accounting for frame time here)
      count2.frameNStart = frameN;  // exact frame index
      
      count2.setAutoDraw(true);
    }

    frameRemains = 3 + 1 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (count2.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      count2.setAutoDraw(false);
    }
    
    // *count1* updates
    if (t >= 4 && count1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      count1.tStart = t;  // (not accounting for frame time here)
      count1.frameNStart = frameN;  // exact frame index
      
      count1.setAutoDraw(true);
    }

    frameRemains = 4 + 1 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (count1.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      count1.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of countdownComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function countdownRoutineEnd() {
  return async function () {
    //------Ending Routine 'countdown'-------
    for (const thisComponent of countdownComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    return Scheduler.Event.NEXT;
  };
}


var practice_trials;
var currentLoop;
function practice_trialsLoopBegin(practice_trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    practice_trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: block_images.length, method: TrialHandler.Method.SEQUENTIAL,
      extraInfo: expInfo, originPath: undefined,
      trialList: undefined,
      seed: undefined, name: 'practice_trials'
    });
    psychoJS.experiment.addLoop(practice_trials); // add the loop to the experiment
    currentLoop = practice_trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisPractice_trial of practice_trials) {
      const snapshot = practice_trials.getSnapshot();
      practice_trialsLoopScheduler.add(importConditions(snapshot));
      practice_trialsLoopScheduler.add(fixation_crossRoutineBegin(snapshot));
      practice_trialsLoopScheduler.add(fixation_crossRoutineEachFrame());
      practice_trialsLoopScheduler.add(fixation_crossRoutineEnd());
      practice_trialsLoopScheduler.add(rt_trialRoutineBegin(snapshot));
      practice_trialsLoopScheduler.add(rt_trialRoutineEachFrame());
      practice_trialsLoopScheduler.add(rt_trialRoutineEnd());
      practice_trialsLoopScheduler.add(feedbackRoutineBegin(snapshot));
      practice_trialsLoopScheduler.add(feedbackRoutineEachFrame());
      practice_trialsLoopScheduler.add(feedbackRoutineEnd());
      practice_trialsLoopScheduler.add(blank_screenRoutineBegin(snapshot));
      practice_trialsLoopScheduler.add(blank_screenRoutineEachFrame());
      practice_trialsLoopScheduler.add(blank_screenRoutineEnd());
      practice_trialsLoopScheduler.add(endLoopIteration(practice_trialsLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}


async function practice_trialsLoopEnd() {
  psychoJS.experiment.removeLoop(practice_trials);

  return Scheduler.Event.NEXT;
}


var test_blocks;
function test_blocksLoopBegin(test_blocksLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    test_blocks = new TrialHandler({
      psychoJS: psychoJS,
      nReps: imgs.length-1, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: undefined,
      seed: undefined, name: 'test_blocks'
    });
    psychoJS.experiment.addLoop(test_blocks); // add the loop to the experiment
    currentLoop = test_blocks;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisTest_block of test_blocks) {
      const snapshot = test_blocks.getSnapshot();
      test_blocksLoopScheduler.add(importConditions(snapshot));
      test_blocksLoopScheduler.add(prepRoutineBegin(snapshot));
      test_blocksLoopScheduler.add(prepRoutineEachFrame());
      test_blocksLoopScheduler.add(prepRoutineEnd());
      test_blocksLoopScheduler.add(countdownRoutineBegin(snapshot));
      test_blocksLoopScheduler.add(countdownRoutineEachFrame());
      test_blocksLoopScheduler.add(countdownRoutineEnd());
      const test_trialsLoopScheduler = new Scheduler(psychoJS);
      test_blocksLoopScheduler.add(test_trialsLoopBegin(test_trialsLoopScheduler, snapshot));
      test_blocksLoopScheduler.add(test_trialsLoopScheduler);
      test_blocksLoopScheduler.add(test_trialsLoopEnd);
      test_blocksLoopScheduler.add(premature_endRoutineBegin(snapshot));
      test_blocksLoopScheduler.add(premature_endRoutineEachFrame());
      test_blocksLoopScheduler.add(premature_endRoutineEnd());
      test_blocksLoopScheduler.add(InterimRoutineBegin(snapshot));
      test_blocksLoopScheduler.add(InterimRoutineEachFrame());
      test_blocksLoopScheduler.add(InterimRoutineEnd());
      test_blocksLoopScheduler.add(endLoopIteration(test_blocksLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}


var test_trials;
function test_trialsLoopBegin(test_trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    test_trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: block_images.length, method: TrialHandler.Method.SEQUENTIAL,
      extraInfo: expInfo, originPath: undefined,
      trialList: undefined,
      seed: undefined, name: 'test_trials'
    });
    psychoJS.experiment.addLoop(test_trials); // add the loop to the experiment
    currentLoop = test_trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisTest_trial of test_trials) {
      const snapshot = test_trials.getSnapshot();
      test_trialsLoopScheduler.add(importConditions(snapshot));
      test_trialsLoopScheduler.add(fixation_crossRoutineBegin(snapshot));
      test_trialsLoopScheduler.add(fixation_crossRoutineEachFrame());
      test_trialsLoopScheduler.add(fixation_crossRoutineEnd());
      test_trialsLoopScheduler.add(rt_trialRoutineBegin(snapshot));
      test_trialsLoopScheduler.add(rt_trialRoutineEachFrame());
      test_trialsLoopScheduler.add(rt_trialRoutineEnd());
      test_trialsLoopScheduler.add(feedbackRoutineBegin(snapshot));
      test_trialsLoopScheduler.add(feedbackRoutineEachFrame());
      test_trialsLoopScheduler.add(feedbackRoutineEnd());
      test_trialsLoopScheduler.add(blank_screenRoutineBegin(snapshot));
      test_trialsLoopScheduler.add(blank_screenRoutineEachFrame());
      test_trialsLoopScheduler.add(blank_screenRoutineEnd());
      test_trialsLoopScheduler.add(endLoopIteration(test_trialsLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}


async function test_trialsLoopEnd() {
  psychoJS.experiment.removeLoop(test_trials);

  return Scheduler.Event.NEXT;
}


async function test_blocksLoopEnd() {
  psychoJS.experiment.removeLoop(test_blocks);

  return Scheduler.Event.NEXT;
}


var fixation_crossComponents;
function fixation_crossRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //------Prepare to start Routine 'fixation_cross'-------
    t = 0;
    fixation_crossClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    routineTimer.add(1.000000);
    // update component parameters for each repeat
    trial_instruction_text_cross.setText(group_instruction2);
    label_a_cross.setText(condition[0]);
    label_l_cross.setText(condition[1]);
    // keep track of which components have finished
    fixation_crossComponents = [];
    fixation_crossComponents.push(trial_instruction_text_cross);
    fixation_crossComponents.push(a_border_cross);
    fixation_crossComponents.push(l_border_cross);
    fixation_crossComponents.push(label_a_cross);
    fixation_crossComponents.push(label_l_cross);
    fixation_crossComponents.push(a_key_cross);
    fixation_crossComponents.push(l_key_cross);
    fixation_crossComponents.push(fixation_cross_text);
    
    for (const thisComponent of fixation_crossComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function fixation_crossRoutineEachFrame() {
  return async function () {
    //------Loop for each frame of Routine 'fixation_cross'-------
    // get current time
    t = fixation_crossClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *trial_instruction_text_cross* updates
    if (t >= 0.0 && trial_instruction_text_cross.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      trial_instruction_text_cross.tStart = t;  // (not accounting for frame time here)
      trial_instruction_text_cross.frameNStart = frameN;  // exact frame index
      
      trial_instruction_text_cross.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (trial_instruction_text_cross.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      trial_instruction_text_cross.setAutoDraw(false);
    }
    
    // *a_border_cross* updates
    if (t >= 0 && a_border_cross.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      a_border_cross.tStart = t;  // (not accounting for frame time here)
      a_border_cross.frameNStart = frameN;  // exact frame index
      
      a_border_cross.setAutoDraw(true);
    }

    frameRemains = 0 + 1 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (a_border_cross.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      a_border_cross.setAutoDraw(false);
    }
    
    // *l_border_cross* updates
    if (t >= 0.0 && l_border_cross.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      l_border_cross.tStart = t;  // (not accounting for frame time here)
      l_border_cross.frameNStart = frameN;  // exact frame index
      
      l_border_cross.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (l_border_cross.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      l_border_cross.setAutoDraw(false);
    }
    
    // *label_a_cross* updates
    if (t >= 0.0 && label_a_cross.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      label_a_cross.tStart = t;  // (not accounting for frame time here)
      label_a_cross.frameNStart = frameN;  // exact frame index
      
      label_a_cross.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (label_a_cross.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      label_a_cross.setAutoDraw(false);
    }
    
    // *label_l_cross* updates
    if (t >= 0.0 && label_l_cross.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      label_l_cross.tStart = t;  // (not accounting for frame time here)
      label_l_cross.frameNStart = frameN;  // exact frame index
      
      label_l_cross.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (label_l_cross.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      label_l_cross.setAutoDraw(false);
    }
    
    // *a_key_cross* updates
    if (t >= 0.0 && a_key_cross.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      a_key_cross.tStart = t;  // (not accounting for frame time here)
      a_key_cross.frameNStart = frameN;  // exact frame index
      
      a_key_cross.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (a_key_cross.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      a_key_cross.setAutoDraw(false);
    }
    
    // *l_key_cross* updates
    if (t >= 0.0 && l_key_cross.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      l_key_cross.tStart = t;  // (not accounting for frame time here)
      l_key_cross.frameNStart = frameN;  // exact frame index
      
      l_key_cross.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (l_key_cross.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      l_key_cross.setAutoDraw(false);
    }
    
    // *fixation_cross_text* updates
    if (t >= 0.0 && fixation_cross_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      fixation_cross_text.tStart = t;  // (not accounting for frame time here)
      fixation_cross_text.frameNStart = frameN;  // exact frame index
      
      fixation_cross_text.setAutoDraw(true);
    }

    frameRemains = 0.0 + 1 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (fixation_cross_text.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      fixation_cross_text.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of fixation_crossComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function fixation_crossRoutineEnd() {
  return async function () {
    //------Ending Routine 'fixation_cross'-------
    for (const thisComponent of fixation_crossComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    return Scheduler.Event.NEXT;
  };
}


var _key_resp_allKeys;
var cutoff;
var trial_type;
var cutoff_translated;
var rt_trialComponents;
function rt_trialRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //------Prepare to start Routine 'rt_trial'-------
    t = 0;
    rt_trialClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    image_rt.setImage(image_path);
    trial_instruction_text_rt.setText(group_instruction2);
    key_resp.keys = undefined;
    key_resp.rt = undefined;
    _key_resp_allKeys = [];
    label_a_rt.setText(condition[0]);
    label_l_rt.setText(condition[1]);
    l_key_rt.setText('L');
    cutoff = imgs[block_i]['cutoff'][trial_i] //< .5 = dunkel; >.5 = hell
    trial_type = imgs[block_i]['trial_type'][trial_i] 
    
    if (cutoff < .5) {
        cutoff_translated = "dark"
    } else {
        cutoff_translated = "bright"
    }
    console.log(block_images)
    //console.log(deadline)
    console.log(image_size)
    // keep track of which components have finished
    rt_trialComponents = [];
    rt_trialComponents.push(image_rt);
    rt_trialComponents.push(trial_instruction_text_rt);
    rt_trialComponents.push(key_resp);
    rt_trialComponents.push(a_border_rt);
    rt_trialComponents.push(l_border_rt);
    rt_trialComponents.push(label_a_rt);
    rt_trialComponents.push(label_l_rt);
    rt_trialComponents.push(a_key_rt);
    rt_trialComponents.push(l_key_rt);
    
    for (const thisComponent of rt_trialComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function rt_trialRoutineEachFrame() {
  return async function () {
    //------Loop for each frame of Routine 'rt_trial'-------
    // get current time
    t = rt_trialClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *image_rt* updates
    if (t >= 0 && image_rt.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      image_rt.tStart = t;  // (not accounting for frame time here)
      image_rt.frameNStart = frameN;  // exact frame index
      
      image_rt.setAutoDraw(true);
    }

    frameRemains = 0 + deadline - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (image_rt.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      image_rt.setAutoDraw(false);
    }
    
    if (image_rt.status === PsychoJS.Status.STARTED){ // only update if being drawn
      image_rt.setSize(image_size, false);
    }
    
    // *trial_instruction_text_rt* updates
    if (t >= 0.0 && trial_instruction_text_rt.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      trial_instruction_text_rt.tStart = t;  // (not accounting for frame time here)
      trial_instruction_text_rt.frameNStart = frameN;  // exact frame index
      
      trial_instruction_text_rt.setAutoDraw(true);
    }

    frameRemains = 0.0 + deadline - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (trial_instruction_text_rt.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      trial_instruction_text_rt.setAutoDraw(false);
    }
    
    // *key_resp* updates
    if (t >= 0.0 && key_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp.tStart = t;  // (not accounting for frame time here)
      key_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp.clearEvents(); });
    }

    frameRemains = 0.0 + deadline - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (key_resp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      key_resp.status = PsychoJS.Status.FINISHED;
  }

    if (key_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp.getKeys({keyList: ['a', 'l'], waitRelease: false});
      _key_resp_allKeys = _key_resp_allKeys.concat(theseKeys);
      if (_key_resp_allKeys.length > 0) {
        key_resp.keys = _key_resp_allKeys[_key_resp_allKeys.length - 1].name;  // just the last key pressed
        key_resp.rt = _key_resp_allKeys[_key_resp_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *a_border_rt* updates
    if (t >= 0.0 && a_border_rt.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      a_border_rt.tStart = t;  // (not accounting for frame time here)
      a_border_rt.frameNStart = frameN;  // exact frame index
      
      a_border_rt.setAutoDraw(true);
    }

    frameRemains = 0.0 + deadline - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (a_border_rt.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      a_border_rt.setAutoDraw(false);
    }
    
    // *l_border_rt* updates
    if (t >= 0.0 && l_border_rt.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      l_border_rt.tStart = t;  // (not accounting for frame time here)
      l_border_rt.frameNStart = frameN;  // exact frame index
      
      l_border_rt.setAutoDraw(true);
    }

    frameRemains = 0.0 + deadline - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (l_border_rt.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      l_border_rt.setAutoDraw(false);
    }
    
    // *label_a_rt* updates
    if (t >= 0.0 && label_a_rt.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      label_a_rt.tStart = t;  // (not accounting for frame time here)
      label_a_rt.frameNStart = frameN;  // exact frame index
      
      label_a_rt.setAutoDraw(true);
    }

    frameRemains = 0.0 + deadline - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (label_a_rt.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      label_a_rt.setAutoDraw(false);
    }
    
    // *label_l_rt* updates
    if (t >= 0.0 && label_l_rt.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      label_l_rt.tStart = t;  // (not accounting for frame time here)
      label_l_rt.frameNStart = frameN;  // exact frame index
      
      label_l_rt.setAutoDraw(true);
    }

    frameRemains = 0.0 + deadline - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (label_l_rt.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      label_l_rt.setAutoDraw(false);
    }
    
    // *a_key_rt* updates
    if (t >= 0.0 && a_key_rt.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      a_key_rt.tStart = t;  // (not accounting for frame time here)
      a_key_rt.frameNStart = frameN;  // exact frame index
      
      a_key_rt.setAutoDraw(true);
    }

    frameRemains = 0.0 + deadline - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (a_key_rt.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      a_key_rt.setAutoDraw(false);
    }
    
    // *l_key_rt* updates
    if (t >= 0.0 && l_key_rt.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      l_key_rt.tStart = t;  // (not accounting for frame time here)
      l_key_rt.frameNStart = frameN;  // exact frame index
      
      l_key_rt.setAutoDraw(true);
    }

    frameRemains = 0.0 + deadline - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (l_key_rt.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      l_key_rt.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of rt_trialComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


var feedback_color;
var feedback_text;
var ISI_length;
function rt_trialRoutineEnd() {
  return async function () {
    //------Ending Routine 'rt_trial'-------
    for (const thisComponent of rt_trialComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // update the trial handler
    if (psychoJS.experiment.currentLoop instanceof MultiStairHandler) {
      psychoJS.experiment.currentLoop.addResponse(key_resp.corr, level);
    }
    psychoJS.experiment.addData('key_resp.keys', key_resp.keys);
    if (typeof key_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp.rt', key_resp.rt);
        routineTimer.reset();
        }
    
    key_resp.stop();
    feedback_color = color_false;
    feedback_text = 'Falsch!';
    
    if (block_i == 0) {
        if (cutoff > .5 & key_resp.keys == hell){
            feedback_text = 'Richtig!';
        } else if (cutoff < .5 & key_resp.keys == dunkel) {
            feedback_text = 'Richtig!';
        }
    } else {
        if (group == "control") {
        in_time_responses += 1;
        if (cutoff > .5 & key_resp.keys == hell) {
            acc_responses +=1;
            feedback_text = '';
        } else if (cutoff < .5 & key_resp.keys == dunkel) {
                acc_responses +=1;
                feedback_text = '';
        } 
    } else {
        if(typeof key_resp.keys === 'undefined'){
            delayed_responses +=1;
            feedback_text = 'Zu langsam!';
        } else {
            in_time_responses += 1;
            feedback_text = '';
            if (cutoff > .5 & key_resp.keys == hell) {
                    acc_responses +=1;
            }else if (cutoff < .5 & key_resp.keys == dunkel) {
                    acc_responses +=1;
            }
        }
    }
    }
    
    ISI_length = 0.3
    
    if(trial_i == block_images.length-1){
        ISI_length = 0
    }
    // the Routine "rt_trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var feedbackComponents;
function feedbackRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //------Prepare to start Routine 'feedback'-------
    t = 0;
    feedbackClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    trial_instruction_text_fb.setText(group_instruction2);
    label_a_fb.setText(condition[0]);
    label_l_fb.setText(condition[1]);
    text_fb.setPos([0, 0]);
    text_fb.setText(feedback_text);
    // keep track of which components have finished
    feedbackComponents = [];
    feedbackComponents.push(trial_instruction_text_fb);
    feedbackComponents.push(a_border_fb);
    feedbackComponents.push(l_border_fb);
    feedbackComponents.push(label_a_fb);
    feedbackComponents.push(label_l_fb);
    feedbackComponents.push(a_key_fb);
    feedbackComponents.push(l_key_fb);
    feedbackComponents.push(text_fb);
    
    for (const thisComponent of feedbackComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function feedbackRoutineEachFrame() {
  return async function () {
    //------Loop for each frame of Routine 'feedback'-------
    // get current time
    t = feedbackClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *trial_instruction_text_fb* updates
    if (t >= 0.0 && trial_instruction_text_fb.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      trial_instruction_text_fb.tStart = t;  // (not accounting for frame time here)
      trial_instruction_text_fb.frameNStart = frameN;  // exact frame index
      
      trial_instruction_text_fb.setAutoDraw(true);
    }

    frameRemains = 0.0 + feedback_length - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (trial_instruction_text_fb.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      trial_instruction_text_fb.setAutoDraw(false);
    }
    
    // *a_border_fb* updates
    if (t >= 0 && a_border_fb.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      a_border_fb.tStart = t;  // (not accounting for frame time here)
      a_border_fb.frameNStart = frameN;  // exact frame index
      
      a_border_fb.setAutoDraw(true);
    }

    frameRemains = 0 + feedback_length - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (a_border_fb.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      a_border_fb.setAutoDraw(false);
    }
    
    // *l_border_fb* updates
    if (t >= 0.0 && l_border_fb.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      l_border_fb.tStart = t;  // (not accounting for frame time here)
      l_border_fb.frameNStart = frameN;  // exact frame index
      
      l_border_fb.setAutoDraw(true);
    }

    frameRemains = 0.0 + feedback_length - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (l_border_fb.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      l_border_fb.setAutoDraw(false);
    }
    
    // *label_a_fb* updates
    if (t >= 0.0 && label_a_fb.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      label_a_fb.tStart = t;  // (not accounting for frame time here)
      label_a_fb.frameNStart = frameN;  // exact frame index
      
      label_a_fb.setAutoDraw(true);
    }

    frameRemains = 0.0 + feedback_length - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (label_a_fb.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      label_a_fb.setAutoDraw(false);
    }
    
    // *label_l_fb* updates
    if (t >= 0.0 && label_l_fb.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      label_l_fb.tStart = t;  // (not accounting for frame time here)
      label_l_fb.frameNStart = frameN;  // exact frame index
      
      label_l_fb.setAutoDraw(true);
    }

    frameRemains = 0.0 + feedback_length - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (label_l_fb.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      label_l_fb.setAutoDraw(false);
    }
    
    // *a_key_fb* updates
    if (t >= 0.0 && a_key_fb.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      a_key_fb.tStart = t;  // (not accounting for frame time here)
      a_key_fb.frameNStart = frameN;  // exact frame index
      
      a_key_fb.setAutoDraw(true);
    }

    frameRemains = 0.0 + feedback_length - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (a_key_fb.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      a_key_fb.setAutoDraw(false);
    }
    
    // *l_key_fb* updates
    if (t >= 0.0 && l_key_fb.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      l_key_fb.tStart = t;  // (not accounting for frame time here)
      l_key_fb.frameNStart = frameN;  // exact frame index
      
      l_key_fb.setAutoDraw(true);
    }

    frameRemains = 0.0 + feedback_length - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (l_key_fb.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      l_key_fb.setAutoDraw(false);
    }
    
    // *text_fb* updates
    if (t >= 0.0 && text_fb.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_fb.tStart = t;  // (not accounting for frame time here)
      text_fb.frameNStart = frameN;  // exact frame index
      
      text_fb.setAutoDraw(true);
    }

    frameRemains = 0.0 + feedback_length - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (text_fb.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      text_fb.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of feedbackComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


var condition_met_acc;
var condition_met_fullscreen;
var condition_met;
var myCompletedURL;
function feedbackRoutineEnd() {
  return async function () {
    //------Ending Routine 'feedback'-------
    for (const thisComponent of feedbackComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('cutoff', block_cutoffs[trial_i]);
    psychoJS.experiment.addData('cutoff_var', cutoff);
    psychoJS.experiment.addData('cutoff_translated', cutoff_translated);
    psychoJS.experiment.addData('image', block_images[trial_i]);
    psychoJS.experiment.addData('session', (parseInt(expInfo["session"])));
    psychoJS.experiment.addData('deadline', deadline);
    psychoJS.experiment.addData('image_size', image_size[0]);
    psychoJS.experiment.addData('group', group);
    
    psychoJS.experiment.addData('feedback_text', feedback_text);
    psychoJS.experiment.addData('block', block_i);
    psychoJS.experiment.addData('trial', trial_i);
    psychoJS.experiment.addData('trial_type', trial_type);
    psychoJS.experiment.addData('Fullscreen', isFullscreen());
    psychoJS.experiment.addData('screen_height', screen_height);
    psychoJS.experiment.addData('height_pixel', $(window).height());
    psychoJS.experiment.addData('width_pixel', $(window).width());
                                                               
    if(block_i != -1){
        if(isFullscreen() != 1){ //i.e., Vollansicht verlassen
            fully_screenys -= 1
        }
        psychoJS.experiment.addData('fully_screenys', fully_screenys);
    } 
    if (block_i > 0){
        psychoJS.experiment.addData('acc_responses', acc_responses);
        psychoJS.experiment.addData('delayed_responses', delayed_responses);
        psychoJS.experiment.addData('in_time_responses', in_time_responses);
    }
    
    condition_met_acc = acc_responses/(in_time_responses + delayed_responses) > 0.5 ? 1 : 0
    condition_met_fullscreen = fully_screenys < 0 ? 0: 1
    
    condition_met = (condition_met_acc + condition_met_fullscreen)
    psychoJS.experiment.addData('condition_met_acc', condition_met_acc);
    psychoJS.experiment.addData('condition_met_fullscreen', condition_met_fullscreen);
    psychoJS.experiment.addData('condition_met', condition_met);
    //myCompletedURL = "https://www.umfragen.uni-kiel.de/index.php/873982?lang=de&pp=" + expInfo["pp"] + "&ext=" 
    myCompletedURL = "https://www.soscisurvey.de/deadline_main_post/?pp=" + expInfo["pp"] + "&ext=" 
    if(condition_met_acc==1){//mehr als50%
        if(condition_met_fullscreen==0){//mal fullscreen verlassen
            myCompletedURL += '42'
        }else{
            myCompletedURL += '3257' //bonus erfolgt
        }
    }else{
        if(condition_met_fullscreen==0){//mal fullscreen verlassen
            myCompletedURL += '2039'
        }else{
            myCompletedURL += '1038'
        }
    }
    
    myCompletedURL += '&rty=0912313'
    
    psychoJS.experiment.addData('myCompletedURL', myCompletedURL);
    // the Routine "feedback" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var blank_screenComponents;
function blank_screenRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //------Prepare to start Routine 'blank_screen'-------
    t = 0;
    blank_screenClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    trial_instruction_text_ISI.setText(group_instruction2);
    label_a3.setText(condition[0]);
    label_l3.setText(condition[1]);
    // keep track of which components have finished
    blank_screenComponents = [];
    blank_screenComponents.push(trial_instruction_text_ISI);
    blank_screenComponents.push(a_border3);
    blank_screenComponents.push(l_border3);
    blank_screenComponents.push(label_a3);
    blank_screenComponents.push(label_l3);
    blank_screenComponents.push(a_key3);
    blank_screenComponents.push(l_key3);
    
    for (const thisComponent of blank_screenComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function blank_screenRoutineEachFrame() {
  return async function () {
    //------Loop for each frame of Routine 'blank_screen'-------
    // get current time
    t = blank_screenClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *trial_instruction_text_ISI* updates
    if (t >= 0.0 && trial_instruction_text_ISI.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      trial_instruction_text_ISI.tStart = t;  // (not accounting for frame time here)
      trial_instruction_text_ISI.frameNStart = frameN;  // exact frame index
      
      trial_instruction_text_ISI.setAutoDraw(true);
    }

    frameRemains = 0.0 + ISI_length - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (trial_instruction_text_ISI.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      trial_instruction_text_ISI.setAutoDraw(false);
    }
    
    // *a_border3* updates
    if (t >= 0 && a_border3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      a_border3.tStart = t;  // (not accounting for frame time here)
      a_border3.frameNStart = frameN;  // exact frame index
      
      a_border3.setAutoDraw(true);
    }

    frameRemains = 0 + ISI_length - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (a_border3.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      a_border3.setAutoDraw(false);
    }
    
    // *l_border3* updates
    if (t >= 0.0 && l_border3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      l_border3.tStart = t;  // (not accounting for frame time here)
      l_border3.frameNStart = frameN;  // exact frame index
      
      l_border3.setAutoDraw(true);
    }

    frameRemains = 0.0 + ISI_length - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (l_border3.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      l_border3.setAutoDraw(false);
    }
    
    // *label_a3* updates
    if (t >= 0.0 && label_a3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      label_a3.tStart = t;  // (not accounting for frame time here)
      label_a3.frameNStart = frameN;  // exact frame index
      
      label_a3.setAutoDraw(true);
    }

    frameRemains = 0.0 + ISI_length - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (label_a3.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      label_a3.setAutoDraw(false);
    }
    
    // *label_l3* updates
    if (t >= 0.0 && label_l3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      label_l3.tStart = t;  // (not accounting for frame time here)
      label_l3.frameNStart = frameN;  // exact frame index
      
      label_l3.setAutoDraw(true);
    }

    frameRemains = 0.0 + ISI_length - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (label_l3.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      label_l3.setAutoDraw(false);
    }
    
    // *a_key3* updates
    if (t >= 0.0 && a_key3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      a_key3.tStart = t;  // (not accounting for frame time here)
      a_key3.frameNStart = frameN;  // exact frame index
      
      a_key3.setAutoDraw(true);
    }

    frameRemains = 0.0 + ISI_length - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (a_key3.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      a_key3.setAutoDraw(false);
    }
    
    // *l_key3* updates
    if (t >= 0.0 && l_key3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      l_key3.tStart = t;  // (not accounting for frame time here)
      l_key3.frameNStart = frameN;  // exact frame index
      
      l_key3.setAutoDraw(true);
    }

    frameRemains = 0.0 + ISI_length - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (l_key3.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      l_key3.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of blank_screenComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function blank_screenRoutineEnd() {
  return async function () {
    //------Ending Routine 'blank_screen'-------
    for (const thisComponent of blank_screenComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    trial_i += 1
    image_path = block_images[trial_i]
    // the Routine "blank_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp3_allKeys;
var instruction3Components;
function instruction3RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //------Prepare to start Routine 'instruction3'-------
    t = 0;
    instruction3Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    info3.setText((((("Sie haben nun die \u00dcbungsdurchg\u00e4nge beendet." + "\n\n") + "In der nachfolgenden Test-Phase werden Sie 4 Bl\u00f6cke mit jeweils ca. 100 Durchg\u00e4ngen bearbeiten. Zwischen den einzelnen Test-Bl\u00f6cken haben Sie jeweils Zeit f\u00fcr eine Pause.") + "\n\n") + group_instruction1));
    key_resp3.keys = undefined;
    key_resp3.rt = undefined;
    _key_resp3_allKeys = [];
    // keep track of which components have finished
    instruction3Components = [];
    instruction3Components.push(header_border3);
    instruction3Components.push(header_text3);
    instruction3Components.push(info3);
    instruction3Components.push(continue_info3);
    instruction3Components.push(continue_border3);
    instruction3Components.push(key_resp3);
    
    for (const thisComponent of instruction3Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function instruction3RoutineEachFrame() {
  return async function () {
    //------Loop for each frame of Routine 'instruction3'-------
    // get current time
    t = instruction3Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *header_border3* updates
    if (t >= 0.0 && header_border3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      header_border3.tStart = t;  // (not accounting for frame time here)
      header_border3.frameNStart = frameN;  // exact frame index
      
      header_border3.setAutoDraw(true);
    }

    
    // *header_text3* updates
    if (t >= 0.0 && header_text3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      header_text3.tStart = t;  // (not accounting for frame time here)
      header_text3.frameNStart = frameN;  // exact frame index
      
      header_text3.setAutoDraw(true);
    }

    
    // *info3* updates
    if (t >= 0.0 && info3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      info3.tStart = t;  // (not accounting for frame time here)
      info3.frameNStart = frameN;  // exact frame index
      
      info3.setAutoDraw(true);
    }

    
    // *continue_info3* updates
    if (t >= 0.0 && continue_info3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_info3.tStart = t;  // (not accounting for frame time here)
      continue_info3.frameNStart = frameN;  // exact frame index
      
      continue_info3.setAutoDraw(true);
    }

    
    // *continue_border3* updates
    if (t >= 0.0 && continue_border3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_border3.tStart = t;  // (not accounting for frame time here)
      continue_border3.frameNStart = frameN;  // exact frame index
      
      continue_border3.setAutoDraw(true);
    }

    
    // *key_resp3* updates
    if (t >= 0.0 && key_resp3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp3.tStart = t;  // (not accounting for frame time here)
      key_resp3.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp3.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp3.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp3.clearEvents(); });
    }

    if (key_resp3.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp3.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp3_allKeys = _key_resp3_allKeys.concat(theseKeys);
      if (_key_resp3_allKeys.length > 0) {
        key_resp3.keys = _key_resp3_allKeys[_key_resp3_allKeys.length - 1].name;  // just the last key pressed
        key_resp3.rt = _key_resp3_allKeys[_key_resp3_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instruction3Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instruction3RoutineEnd() {
  return async function () {
    //------Ending Routine 'instruction3'-------
    for (const thisComponent of instruction3Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // update the trial handler
    if (psychoJS.experiment.currentLoop instanceof MultiStairHandler) {
      psychoJS.experiment.currentLoop.addResponse(key_resp3.corr, level);
    }
    psychoJS.experiment.addData('key_resp3.keys', key_resp3.keys);
    if (typeof key_resp3.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp3.rt', key_resp3.rt);
        routineTimer.reset();
        }
    
    key_resp3.stop();
    // the Routine "instruction3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp4_allKeys;
var instruction4Components;
function instruction4RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //------Prepare to start Routine 'instruction4'-------
    t = 0;
    instruction4Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    key_resp4.keys = undefined;
    key_resp4.rt = undefined;
    _key_resp4_allKeys = [];
    info4.setText((((((((((("Legen Sie nun bitte Ihre beiden Zeigefinger auf den Tasten \"A\" (\"" + condition[0]) + "\") und \"L\" (\"") + condition[1]) + "\") ab.") + "\nEs ist wichtig, dass Sie Ihre beiden Zeigefinger w\u00e4hren aller Durchg\u00e4nge eines jeden Test-Blocks auf diesen Tasten liegen lassen.") + "\n\n") + "Achtung:") + "\n") + "Sie werden fortan nur noch dann R\u00fcckmeldung erhalten, ") + group_instruction3));
    // keep track of which components have finished
    instruction4Components = [];
    instruction4Components.push(header_border4);
    instruction4Components.push(header_text4);
    instruction4Components.push(continue_info4);
    instruction4Components.push(continue_border4);
    instruction4Components.push(key_resp4);
    instruction4Components.push(info4);
    
    for (const thisComponent of instruction4Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function instruction4RoutineEachFrame() {
  return async function () {
    //------Loop for each frame of Routine 'instruction4'-------
    // get current time
    t = instruction4Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *header_border4* updates
    if (t >= 0.0 && header_border4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      header_border4.tStart = t;  // (not accounting for frame time here)
      header_border4.frameNStart = frameN;  // exact frame index
      
      header_border4.setAutoDraw(true);
    }

    
    // *header_text4* updates
    if (t >= 0.0 && header_text4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      header_text4.tStart = t;  // (not accounting for frame time here)
      header_text4.frameNStart = frameN;  // exact frame index
      
      header_text4.setAutoDraw(true);
    }

    
    // *continue_info4* updates
    if (t >= 0.0 && continue_info4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_info4.tStart = t;  // (not accounting for frame time here)
      continue_info4.frameNStart = frameN;  // exact frame index
      
      continue_info4.setAutoDraw(true);
    }

    
    // *continue_border4* updates
    if (t >= 0.0 && continue_border4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_border4.tStart = t;  // (not accounting for frame time here)
      continue_border4.frameNStart = frameN;  // exact frame index
      
      continue_border4.setAutoDraw(true);
    }

    
    // *key_resp4* updates
    if (t >= 0.0 && key_resp4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp4.tStart = t;  // (not accounting for frame time here)
      key_resp4.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp4.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp4.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp4.clearEvents(); });
    }

    if (key_resp4.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp4.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp4_allKeys = _key_resp4_allKeys.concat(theseKeys);
      if (_key_resp4_allKeys.length > 0) {
        key_resp4.keys = _key_resp4_allKeys[_key_resp4_allKeys.length - 1].name;  // just the last key pressed
        key_resp4.rt = _key_resp4_allKeys[_key_resp4_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *info4* updates
    if (t >= 0.0 && info4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      info4.tStart = t;  // (not accounting for frame time here)
      info4.frameNStart = frameN;  // exact frame index
      
      info4.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of instruction4Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function instruction4RoutineEnd() {
  return async function () {
    //------Ending Routine 'instruction4'-------
    for (const thisComponent of instruction4Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // update the trial handler
    if (psychoJS.experiment.currentLoop instanceof MultiStairHandler) {
      psychoJS.experiment.currentLoop.addResponse(key_resp4.corr, level);
    }
    psychoJS.experiment.addData('key_resp4.keys', key_resp4.keys);
    if (typeof key_resp4.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp4.rt', key_resp4.rt);
        routineTimer.reset();
        }
    
    key_resp4.stop();
    // the Routine "instruction4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp_interim_allKeys;
var InterimComponents;
function InterimRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //------Prepare to start Routine 'Interim'-------
    t = 0;
    InterimClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    header_text_interim.setText(interim_header);
    text_interim.setText(interim);
    continue_border_interim.setText(interim_footer);
    key_resp_interim.keys = undefined;
    key_resp_interim.rt = undefined;
    _key_resp_interim_allKeys = [];
    dat = new Date();
    date_within = dat.getFullYear() +"-"+ (dat.getMonth()+1) +"-"+ dat.getDate() +" "+ dat.getHours() +":"+ dat.getMinutes() +":"+ dat.getSeconds();
    psychoJS.experiment.addData("date_within", date_within);
    
    // keep track of which components have finished
    InterimComponents = [];
    InterimComponents.push(header_border_interim);
    InterimComponents.push(header_text_interim);
    InterimComponents.push(text_interim);
    InterimComponents.push(continue_interim);
    InterimComponents.push(continue_border_interim);
    InterimComponents.push(key_resp_interim);
    
    for (const thisComponent of InterimComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function InterimRoutineEachFrame() {
  return async function () {
    //------Loop for each frame of Routine 'Interim'-------
    // get current time
    t = InterimClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *header_border_interim* updates
    if (t >= 0.0 && header_border_interim.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      header_border_interim.tStart = t;  // (not accounting for frame time here)
      header_border_interim.frameNStart = frameN;  // exact frame index
      
      header_border_interim.setAutoDraw(true);
    }

    
    // *header_text_interim* updates
    if (t >= 0.0 && header_text_interim.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      header_text_interim.tStart = t;  // (not accounting for frame time here)
      header_text_interim.frameNStart = frameN;  // exact frame index
      
      header_text_interim.setAutoDraw(true);
    }

    
    // *text_interim* updates
    if (t >= 0.0 && text_interim.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_interim.tStart = t;  // (not accounting for frame time here)
      text_interim.frameNStart = frameN;  // exact frame index
      
      text_interim.setAutoDraw(true);
    }

    
    // *continue_interim* updates
    if (t >= 0.0 && continue_interim.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_interim.tStart = t;  // (not accounting for frame time here)
      continue_interim.frameNStart = frameN;  // exact frame index
      
      continue_interim.setAutoDraw(true);
    }

    
    // *continue_border_interim* updates
    if (t >= 0.0 && continue_border_interim.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_border_interim.tStart = t;  // (not accounting for frame time here)
      continue_border_interim.frameNStart = frameN;  // exact frame index
      
      continue_border_interim.setAutoDraw(true);
    }

    
    // *key_resp_interim* updates
    if (t >= 0.0 && key_resp_interim.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_interim.tStart = t;  // (not accounting for frame time here)
      key_resp_interim.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_interim.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_interim.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_interim.clearEvents(); });
    }

    if (key_resp_interim.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_interim.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_interim_allKeys = _key_resp_interim_allKeys.concat(theseKeys);
      if (_key_resp_interim_allKeys.length > 0) {
        key_resp_interim.keys = _key_resp_interim_allKeys[_key_resp_interim_allKeys.length - 1].name;  // just the last key pressed
        key_resp_interim.rt = _key_resp_interim_allKeys[_key_resp_interim_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of InterimComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


var d;
var date_end;
function InterimRoutineEnd() {
  return async function () {
    //------Ending Routine 'Interim'-------
    for (const thisComponent of InterimComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // update the trial handler
    if (psychoJS.experiment.currentLoop instanceof MultiStairHandler) {
      psychoJS.experiment.currentLoop.addResponse(key_resp_interim.corr, level);
    }
    psychoJS.experiment.addData('key_resp_interim.keys', key_resp_interim.keys);
    if (typeof key_resp_interim.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_interim.rt', key_resp_interim.rt);
        routineTimer.reset();
        }
    
    key_resp_interim.stop();
    dat = new Date();
    date_within = dat.getFullYear() +"-"+ (dat.getMonth()+1) +"-"+ dat.getDate() +" "+ dat.getHours() +":"+ dat.getMinutes() +":"+ dat.getSeconds();
    psychoJS.experiment.addData("date_within", date_within);
    
    if(block_i >= 4){
        d = new Date();
        date_end = d.getFullYear() +"-"+ (d.getMonth()+1) +"-"+ d.getDate() +" "+ d.getHours() +":"+ d.getMinutes() +":"+ d.getSeconds();
        psychoJS.experiment.addData("date_end", date_end);
    }
    // the Routine "Interim" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


function endLoopIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        const thisTrial = snapshot.getCurrentTrial();
        if (typeof thisTrial === 'undefined' || !('isTrials' in thisTrial) || thisTrial.isTrials) {
          psychoJS.experiment.nextEntry(snapshot);
        }
      }
    return Scheduler.Event.NEXT;
    }
  };
}


function importConditions(currentLoop) {
  return async function () {
    psychoJS.importAttributes(currentLoop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}


async function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  
  
  
  
  
  
  
  
  
  
  
  
  
  psychoJS.setRedirectUrls(myCompletedURL, 'about:blank');
  
  
  
  
  
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
