function extractReviewData(currentRevObj) {
   var peopleReactions = currentRevObj.children[0].innerHTML.split("<br>")
   var hasFunnyProp = true;
   var hasHelpfulProp = true;
   if(peopleReactions.length == 1 && currentRevObj.children[0].innerHTML.indexOf("funny") == -1) {
      hasFunnyProp = false;
   } 
   else if(peopleReactions.length == 1 && currentRevObj.children[0].innerHTML.indexOf("helpful") == -1) {
      hasHelpfulProp = false;
   }
   
   var percentFoundHelpful;
   var totalPeople;
   if(!hasHelpfulProp) {
      percentFoundHelpful = 0.0;
   }
   else {
      var helpful = peopleReactions[0].split("people").join("").split("person")[0].split("of")
      if(helpful.length == 1) {
         percentFoundHelpful = 1
         totalPeople = parseInt(helpful[0].replace(/\,/g, ''));
      }
      else {
         var foundHelpful = parseInt(helpful[0].replace(/\,/g, ''));
         totalPeople = parseInt(helpful[1].replace(/\,/g, ''));

         percentFoundHelpful = foundHelpful/(totalPeople*1.0)
      }
   }

   var percentFoundFunny;
   if(peopleReactions.length == 2) {
      var foundFunny = parseInt(peopleReactions[1].split(" ")[0].replace(/\,/g, ''))
      percentFoundFunny = foundFunny/(totalPeople*1.0);
   }
   else if (peopleReactions.length == 1 && !hasFunnyProp){
      percentFoundFunny = 0;
   }
   else {
      percentFoundFunny = 1;
   }

   var dataOnHours = currentRevObj.children[1].innerText.split(" ")
   var actualHours = parseFloat(dataOnHours[dataOnHours.length - 4].split("\n")[6-dataOnHours.length].replace(/\,/g, ''))
   var actualTextArr = currentRevObj.children[2].innerText.split("\n")
   actualTextArr.shift(1);
   var actualText = actualTextArr.join(" ")

   var finalJsonObj = {};
   finalJsonObj["review"] = actualText
   finalJsonObj["hours"] = actualHours
   finalJsonObj["funny_percent"] = percentFoundFunny
   finalJsonObj["helpful_percent"] = percentFoundHelpful
   return finalJsonObj;
}


function collectReviews(typesOfReviews)  {
   var reviews = [];
   var index = 0;
   while(reviews.length != 50) {
      var tempRev = document.getElementsByClassName("apphub_UserReviewCardContent")[index];
      if(tempRev.children[0].innerHTML.indexOf("No ratings yet") == -1 && tempRev.children[1].innerText.indexOf("\n" + typesOfReviews) != -1) {
         reviews.push(extractReviewData(tempRev));
      }
      index++;
   }
   return reviews;
}

function saveTextAsFile(textToWrite, fileNameToSaveAs)
{
    var textFileAsBlob = new Blob([textToWrite], {type:'text/plain'});
      var downloadLink = document.createElement("a");
    downloadLink.download = fileNameToSaveAs;
    downloadLink.innerHTML = "Download File";
    if (window.webkitURL != null)
    {
        // Chrome allows the link to be clicked
        // without actually adding it to the DOM.
        downloadLink.href = window.webkitURL.createObjectURL(textFileAsBlob);
    }
    else
    {
        // Firefox requires the link to be added to the DOM
        // before it can be clicked.
        downloadLink.href = window.URL.createObjectURL(textFileAsBlob);
        downloadLink.onclick = destroyClickedElement;
        downloadLink.style.display = "none";
        document.body.appendChild(downloadLink);
    }

    downloadLink.click();
}
