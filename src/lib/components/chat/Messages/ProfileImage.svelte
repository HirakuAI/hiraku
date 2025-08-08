<script lang="ts">
	import { HIRAKU_BASE_URL } from '$lib/constants';

	export let className = 'size-8';
	export let src = `${HIRAKU_BASE_URL}/static/favicon.png`;
</script>

<img
	crossorigin="anonymous"
	src={src === ''
		? `${HIRAKU_BASE_URL}/static/favicon.png`
		: src.startsWith(HIRAKU_BASE_URL) ||
			  src.startsWith('https://www.gravatar.com/avatar/') ||
			  src.startsWith('data:') ||
			  src.startsWith('/')
			? src
			: `/user.png`}
	class=" {className} object-cover rounded-full -translate-y-[1px]"
	alt="profile"
	draggable="false"
/>
