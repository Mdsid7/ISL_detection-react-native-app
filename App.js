import React, { useState, useEffect } from 'react';
import { Button, StyleSheet, Text, Image, View, Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native';
//import * as jpeg from 'jpeg-js';
import * as FileSystem from 'expo-file-system';
import * as ImageManipulator  from 'expo-image-manipulator';

export default function ISL_detection() {
  const [image, setImage] = useState(null);
  const [islDetector,setIslDetector]=useState(null)
 
  const [displayText,setDisplayText]=useState('')
  
  
  useEffect(() => {
    (async () => {
      if (Platform.OS !== 'web') {
        const { status } = await ImagePicker.requestCameraRollPermissionsAsync();
        if (status !== 'granted') {
          alert('Sorry, we need camera roll permissions to make this work!');
        }
      }
    })();
  }, []);
  useEffect(() => {
    async function loadModel(){
      console.log("[+] Application started")
      setDisplayText("[+] Application started")
      //Wait for tensorflow module to be ready
      const tfReady = await tf.ready();
      console.log("[+] Loading custom ISL detection model")
      setDisplayText("[+] Loading custom ISL detection model")
     //Loading own custom trained model
      const modelJson = await require("./assets/my_model/model.json");
      const modelWeights = await require("./assets/my_model/modelWeights.bin");
      
      const islDetector = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
      
      setIslDetector(islDetector)
     
      console.log("[+] Model Loaded")
      setDisplayText("[+] Model Loaded")

    }
    loadModel()
  }, []); 
  
  
  /*function imageToTensor(rawImageData){
    
    //Function to convert jpeg image to tensors
    const TO_UINT8ARRAY = true;
    const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);
    // Drop the alpha channel info for mobilenet
    const buffer = new Uint8Array(width * height * 3);
    let offset = 0; // offset into original data
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset];
      buffer[i + 1] = data[offset + 1];
      buffer[i + 2] = data[offset + 2];
      offset += 4;
    }
    return tf.tensor3d(buffer, [height, width, 3]);
  }*/
   
  const getPredict = async() => {
      
    
      console.log("[+] Retrieving image from link :"+image)
       
         let fileUri = image; 
        const imgB64 = await FileSystem.readAsStringAsync(fileUri, {
	       encoding: FileSystem.EncodingType.Base64, });
        const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
        const rawImageData = new Uint8Array(imgBuffer)
        const imageTensor = decodeJpeg(rawImageData).reshape([1,224,224,3])
       
        const winner = tf.clone(imageTensor)
        const abc = tf.cast(winner,'float32') ;
        const yes = abc.div(255.0)
        const classes = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        const pred = await islDetector.predict(yes)
        const predictions = classes[tf.argMax(pred,-1).dataSync()[0]];
        console.log(predictions)
        setDisplayText(predictions)
        console.log("[+] Prediction Completed")
    }

  const openCamera = async () => {
    let result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.All,
      allowsEditing: false,
      aspect: [4, 3],
      quality: 1,
    });
    const manipResult = await ImageManipulator.manipulateAsync(
      result.localUri || result.uri,
      [{ resize: { width: 224, height: 224 } }],
         
    );  

    console.log(manipResult);

    if (!manipResult.cancelled) {
      setImage(manipResult.uri);
    }
  };

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.All,
      allowsEditing: false,
      aspect: [4, 3],
      quality: 1,  
    });
    const manipResult = await ImageManipulator.manipulateAsync(
      result.localUri || result.uri,
      [{ resize: { width: 224, height: 224 } }],
    );  

    console.log(manipResult);

    if (!manipResult.cancelled) {
      setImage(manipResult.uri);
    }
  };
  
 
  return (
    <>
    <View style={styles.container}></View>
    <Text style={styles.red}>Indian Sign Language</Text>
    <View style={styles.container2}>
    {image && <Image source={{ uri: image }} style={{ width: 270, height: 240 }} />}
    </View>
    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
    
    <Text style={{ fontSize: 30, color:'green', fontWeight:'bold'}}>{displayText} </Text> 
  
    <View style={styles.container}></View>
    <Button 
          title="Predict"
          onPress={()=>{getPredict()}}
        />  
    <View style={styles.container}></View>
    <Button title="Open Camera" onPress={openCamera} />
    <View style={styles.container}></View>
    <Button title="Upload from camera roll" onPress={pickImage} />   
    </View>
  </>
  
  );
}

const styles = StyleSheet.create({
  container: {
    marginTop: 50,
  },
  container2: {
    marginTop: 30,
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  red: {
    color: 'white',
    fontWeight: 'bold',
    backgroundColor: 'rgb(128, 0, 0)',
    textAlignVertical:'center',
    fontStyle:'italic',
    textAlign:'center',
    minHeight:80,
    fontSize: 30,
    
  },
  text: {
    color: '#000000',
    fontSize: 16
  },
});
