



async function runExample() {
    

  // Create an ONNX inference session with default backend.
  const session = new onnx.InferenceSession();
 
  await session.loadModel("./F1plusF2_CFDracewaySIO.onnx");

  var x = new Float32Array(1, 4);

 
  

  var x = [];
  x[0] = document.getElementById('box0').value;
  x[1] = document.getElementById('box1').value;
  x[2] = document.getElementById('box2').value;
  x[3] = document.getElementById('box3').value;

 

  const tensorX = new onnx.Tensor(x, 'float32', [1, 4]);
  
   
  
  const outputMap = await session.run([tensorX]);
  const outputData = outputMap.get('output1');
  
  
 

  // PREDS DIV 
  const predictions = document.getElementById('predictions');
  
 


  predictions.innerHTML = `<hr> Got an output Tensor of size ${outputData.data.length} with values being: <br/> 
  <table>
  <tr><td> o_raceway_flame_temp_k</td><td> ${outputData.data[0]} </td></tr>
  <tr><td> o_raceway_coal_burn_perce</td><td> ${outputData.data[1]} </td></tr> 
  <tr><td> o_raceway_volume_m</td><td> ${outputData.data[2]} </td></tr>
  <tr><td> o_raceway_depth_m</td>   <td> ${outputData.data[3]} </td></tr> 
 </table>   `;
 



}
