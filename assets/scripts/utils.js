function sizeBanner() {
	// var card = document.getElementsByClassName("syllabus-entry")[0];

	// for (let banner of document.getElementsByClassName("entry-banner__img")) {
	// 	if (card.offsetWidth > card.offsetHeight) {
	// 		banner.width  = card.offsetWidth;
	// 		banner.height = card.offsetWidth / 16 * 9;
	// 	} else {
	// 		banner.height = card.offsetHeight;
	// 		banner.width  = card.offsetHeight / 9 * 16;
	// 	}
	// };
}

// function hideme() {
//     var btns = {};
//     btns.yt = document.getElementsByClassName("btn-youtube");
//     btns.fb = document.getElementsByClassName("btn-facebook");

//     for (var yt of btns.yt) {
//         yt.style.visibility = (yt.href === document.URL) ? "hidden" : "visibile";
//     }
//     for (var fb of btns.fb) {
//         fb.style.visibility = (fb.href === document.URL) ? "hidden" : "visibile";
//   }
// }

// function breadcrumb(page) {
// 	var subsites = ["hackpack", "course", "data-science", "intelligence"];

// 	subsites.forEach(function (subsite) {
// 		if (page.includes(subsite)) {
// 			document.getElementById("sigai-group-brand").classList.add("active")
// 			document.getElementById("sigai-group-breadcrumb").innerHTML = `/${subsite}`;
// 		}
// 	});
// }

// function highlightLinks() {
// 	var page = window.location.href;
// 	breadcrumb(page);

// 	var nav = document.getElementById("navbarGlobal").firstElementChild;

// 	for (let ele of nav.children)
// 		if (page.includes(ele.href))
// 			ele.classList.add("active");
// }