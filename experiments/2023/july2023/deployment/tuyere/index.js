
      

async function runExample1() {
    
           ////////////////////////////////////////////////
      
           var x = new Float32Array(1, 6);

           var x = [];
           x[0] = document.getElementById('box0c1').value;
           x[1] = document.getElementById('box1c1').value;
           x[2] = document.getElementById('box2c1').value;
           x[3] = document.getElementById('box3c1').value;
           x[4] = document.getElementById('box4c1').value;
           x[5] = document.getElementById('box5c1').value;
     
           const tensorX = new ort.Tensor('float32', x, [1, 6]);
           const feeds = { float_input: tensorX};
           
           
           
           const session1 = await ort.InferenceSession.create('./xgboost_tuyere_t_k_ort.onnx');
           const results1 = await session1.run(feeds);
           const outputData1 = results1.variable.data;
      
          
      
           const session2 = await ort.InferenceSession.create('./xgboost_tuyere_exit_velo_m_s_ort.onnx');
           const results2 = await session2.run(feeds);
           const outputData2 = results2.variable.data;
          


  // PREDS DIV 
  const predictions = document.getElementById('predictions1');
  

 predictions.innerHTML = `<hr> Got an output Tensor with values being: <br/> 
 <table>
 
  <tr>
  <td> o_tuyere_t_k</td>
  <td id="c1td0"> ${outputData1} </td>
  </tr>
  
  <tr>
  <td> o_tuyere_exit_velo_m_s </td>
  <td id="c1td1"> ${outputData2} </td>
  </tr> 
  
 </table>   `;
 

runDiff();

}


async function runExample2() {
           
           var x = new Float32Array(1, 6);

           var x = [];
           x[0] = document.getElementById('box0c2').value;
           x[1] = document.getElementById('box1c2').value;
           x[2] = document.getElementById('box2c2').value;
           x[3] = document.getElementById('box3c2').value;
           x[4] = document.getElementById('box4c2').value;
           x[5] = document.getElementById('box5c2').value;
           

           const tensorX = new ort.Tensor('float32', x, [1, 6]);
           const feeds = { float_input: tensorX};
           
           
           const session1 = await ort.InferenceSession.create('./xgboost_tuyere_t_k_ort.onnx');
           const results1 = await session1.run(feeds);
           const outputData1 = results1.variable.data;

 

           const session2 = await ort.InferenceSession.create('./xgboost_tuyere_exit_velo_m_s_ort.onnx');
           const results2 = await session2.run(feeds);
           const outputData2 = results2.variable.data;
          
 
 

  // PREDS DIV 
  const predictions = document.getElementById('predictions2');
  

  predictions.innerHTML = `<hr> Got an output Tensor  with values being: <br/> 
 <table>
 
  <tr>
  <td> o_tuyere_t_k</td>
  <td id="c2td0"> ${outputData1} </td>
  </tr>
  
  <tr>
  <td> o_tuyere_exit_velo_m_s </td>
  <td id="c2td1"> ${outputData2} </td>
  </tr> 
   
 </table>   `;
 

runDiff();

}


async function runDiff() {
    
    var c1td0 = parseFloat( document.getElementById('c1td0').innerHTML );
    var c1td1 = parseFloat( document.getElementById('c1td1').innerHTML );
 
    
    var c2td0 = parseFloat( document.getElementById('c2td0').innerHTML );
    var c2td1 = parseFloat( document.getElementById('c2td1').innerHTML );
   
    
    td0 = c1td0 - c2td0;
    td1 = c1td1 - c2td1;
 

     difference.innerHTML = `<hr> Difference is: <br/> 
     
 <table>
 
  <tr>
  <td> o_tuyere_t_k</td>
  <td> ${td0.toFixed(2)} </td>
  </tr>
  
  <tr>
  <td> o_tuyere_exit_velo_m_s </td>
  <td> ${td1.toFixed(2)} </td>
  </tr> 
  

  
 </table>   `;
    
}

