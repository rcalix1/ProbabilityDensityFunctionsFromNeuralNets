



async function runExample1() {
    

  // Create an ONNX inference session with default backend.
  const session = new onnx.InferenceSession();
 
  await session.loadModel("./s536NewShaftPDFshapingRC.onnx");

  var x = new Float32Array(1, 6);

 
  

  var x = [];
  x[0] = document.getElementById('box0c1').value;
  x[1] = document.getElementById('box1c1').value;
  x[2] = document.getElementById('box2c1').value;
  x[3] = document.getElementById('box3c1').value;
  x[4] = document.getElementById('box4c1').value;
  x[5] = document.getElementById('box5c1').value;
 

  
  const tensorX = new onnx.Tensor(x, 'float32', [1, 6]);
  
   
  
  const outputMap = await session.run([tensorX]);
  const outputData = outputMap.get('output1');
  
  
 

  // PREDS DIV 
  const predictions = document.getElementById('predictions1');
  
 


  predictions.innerHTML = `<hr> Got an output Tensor of size ${outputData.data.length} with values being: <br/> 
 <table>
  <tr>
  <td> o_shaft_top_gas_temp_c</td>
  <td id="c1td0"> ${outputData.data[0].toFixed(2)} </td>
  </tr>
  <tr>
  <td> o_shaft_press_drop_pa</td>
  <td id="c1td1"> ${outputData.data[1].toFixed(2)} </td>
  </tr> 
  <tr><td> o_shaft_coke_rate_kg_thm</td>
  <td id="c1td2"> ${outputData.data[2].toFixed(2)} </td>
  </tr>
  <tr>
  <td> o_shaft_co_v_perc</td>
  <td id="c1td3"> ${outputData.data[3].toFixed(2)} </td>
  </tr> 
  <tr>
  <td> o_shaft_co2_v_perc</td>
  <td id="c1td4"> ${outputData.data[4].toFixed(2)} </td>
  </tr> 
 </table>   `;
 

runDiff();

}


async function runExample2() {
    

  // Create an ONNX inference session with default backend.
  const session = new onnx.InferenceSession();
 
  await session.loadModel("./s536NewShaftPDFshapingRC.onnx");

  var x = new Float32Array(1, 6);

 
  

  var x = [];
  x[0] = document.getElementById('box0c2').value;
  x[1] = document.getElementById('box1c2').value;
  x[2] = document.getElementById('box2c2').value;
  x[3] = document.getElementById('box3c2').value;
  x[4] = document.getElementById('box4c2').value;
  x[5] = document.getElementById('box5c2').value;
 

  
  const tensorX = new onnx.Tensor(x, 'float32', [1, 6]);
  
   
  
  const outputMap = await session.run([tensorX]);
  const outputData = outputMap.get('output1');
  
  
 

  // PREDS DIV 
  const predictions = document.getElementById('predictions2');
  
 


  predictions.innerHTML = `<hr> Got an output Tensor of size ${outputData.data.length} with values being: <br/> 
 <table>
  <tr>
  <td> o_shaft_top_gas_temp_c</td>
  <td id="c2td0"> ${outputData.data[0].toFixed(2)} </td>
  </tr>
  <tr>
  <td>o_shaft_press_drop_pa</td>
  <td id="c2td1"> ${outputData.data[1].toFixed(2)} </td>
  </tr> 
  <tr><td> o_shaft_coke_rate_kg_thm</td>
  <td id="c2td2"> ${outputData.data[2].toFixed(2)} </td>
  </tr>
  <tr>
  <td> o_shaft_co_v_perc</td>
  <td id="c2td3"> ${outputData.data[3].toFixed(2)} </td>
  </tr> 
  <tr>
  <td> o_shaft_co2_v_perc</td>
  <td id="c2td4"> ${outputData.data[4].toFixed(2)} </td>
  </tr> 
 </table>   `;
 

runDiff();

}


async function runDiff() {
    
    var c1td0 = parseFloat( document.getElementById('c1td0').innerHTML );
    var c1td1 = parseFloat( document.getElementById('c1td1').innerHTML );
    var c1td2 = parseFloat( document.getElementById('c1td2').innerHTML );
    var c1td3 = parseFloat( document.getElementById('c1td3').innerHTML );
    var c1td4 = parseFloat( document.getElementById('c1td4').innerHTML );
    
    var c2td0 = parseFloat( document.getElementById('c2td0').innerHTML );
    var c2td1 = parseFloat( document.getElementById('c2td1').innerHTML );
    var c2td2 = parseFloat( document.getElementById('c2td2').innerHTML );
    var c2td3 = parseFloat( document.getElementById('c2td3').innerHTML );
    var c2td4 = parseFloat( document.getElementById('c2td4').innerHTML );
    
    td0 = c1td0 - c2td0;
    td1 = c1td1 - c2td1;
    td2 = c1td2 - c2td2;
    td3 = c1td3 - c2td3;
    td4 = c1td4 - c2td4;

 
     difference.innerHTML = `<hr> Difference is: <br/> 
 <table>
  <tr>
  <td> o_shaft_top_gas_temp_c</td>
  <td> ${td0.toFixed(2)} </td>
  </tr>
  <tr>
  <td> o_shaft_press_drop_pa</td>
  <td> ${td1.toFixed(2)} </td>
  </tr> 
  <tr><td> o_shaft_coke_rate_kg_thm</td>
  <td> ${td2.toFixed(2)} </td>
  </tr>
  <tr>
  <td> o_shaft_co_v_perc</td>
  <td> ${td3.toFixed(2)} </td>
  </tr> 
  <tr>
  <td> o_shaft_co2_v_perc</td>
  <td> ${td4.toFixed(2)} </td>
  </tr> 
 </table>   `;
    
}

