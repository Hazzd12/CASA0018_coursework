import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';


void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: AudioRecorder(),
    );
  }
}

class AudioRecorder extends StatefulWidget {
  @override
  _AudioRecorderState createState() => _AudioRecorderState();
}

class _AudioRecorderState extends State<AudioRecorder> {
  FlutterSoundRecorder? _recorder;
  bool _isRecording = false;
  String _selectedAnimal = 'Rooster';
  final List<String> _animals = ['Rooster', 'Dog', 'Pig', 'Cow', 'Frog'];

  @override
  void initState() {
    super.initState();
    _initRecorder();
  }

  Future<void> _initRecorder() async {
    _recorder = FlutterSoundRecorder();
    final status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      throw RecordingPermissionException('Microphone permission not granted');
    }
    await _recorder!.openRecorder();
  }

  Future<void> _startRecording() async {
    final directory = await getApplicationDocumentsDirectory();

    final path = '${directory.path}/$_selectedAnimal.wav';


    await _recorder!.startRecorder(toFile: path);
    setState(() => _isRecording = true);

    // Auto stop after 2 seconds
    Future.delayed(Duration(seconds: 2), () {
      _stopRecording(path);
    });
  }

  Future<void> _stopRecording(String path) async {
    await _recorder!.stopRecorder();
    setState(() => _isRecording = false);
    _uploadFile(path);
  }

  Future<void> _uploadFile(String path) async {
    final url = 'http://10.129.118.189:8080/upload';
    final request = http.MultipartRequest('POST', Uri.parse(url))
      ..files.add(await http.MultipartFile.fromPath('file', path));
    final response = await request.send();
    final responseData = await response.stream.bytesToString();
    _showResponse(responseData);
  }


  void _showResponse(String response) {

    Map<String, dynamic> jsonData = json.decode(response); // 将JSON字符串解析为Map


    dynamic score = jsonData['Your score'];

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        content: Text('Your Score: ${score}'),
        actions: [
          TextButton(
            child: Text('OK'),
            onPressed: () => Navigator.of(context).pop(),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Game of imitating animal sounds')),
      body: Center(
    child:Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: <Widget>[
          DropdownButton<String>(
            itemHeight: 50,
            value: _selectedAnimal,
            onChanged: (String? newValue) {
              setState(() {
                _selectedAnimal = newValue!;
              });
            },
            items: _animals.map<DropdownMenuItem<String>>((String value) {
              return DropdownMenuItem<String>(
                value: value,
                child: Text(value),
              );
            }).toList(),
          ),
          SizedBox(height: 200,),
          ElevatedButton(
            onPressed: _isRecording ? null : _startRecording,
            child: Text(_isRecording ? 'Recording...' : 'Record'),
          ),
        ],
      ),
    ));
  }

  @override
  void dispose() {
    _recorder?.closeRecorder();
    super.dispose();
  }
}
